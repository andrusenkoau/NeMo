# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cache-aware streaming inference for unified Conformer RNNT models.

This script is intended for unified RNNT checkpoints trained with
``att_context_style=chunked_limited_with_rc``. It configures the encoder with a
streaming [left, chunk, right] context, then decodes using the Conformer cache
path so the left context is reused instead of recomputed for every chunk.

Example:

python speech_to_text_cache_aware_streaming_infer_rnnt.py \
    model_path=unified_rnnt.nemo \
    dataset_manifest=manifest.json \
    output_manifest=preds.json \
    left_context_secs=5.6 \
    chunk_secs=0.48 \
    right_context_secs=0.56 \
    batch_size=16
"""

import glob
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.manifest_utils import filepath_to_absolute, read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, batched_hyps_to_hypotheses
from nemo.collections.asr.parts.utils.streaming_utils import AudioBatch, ContextSize, SimpleAudioDataset, StreamingBatchedAudioBuffer
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging


def make_divisible_by(num, factor: int) -> int:
    """Make num divisible by factor."""
    return (num // factor) * factor


@dataclass
class TranscriptionConfig:
    """Transcription configuration for unified cache-aware RNNT inference."""

    # Required configs
    model_path: Optional[str] = None
    pretrained_name: Optional[str] = None
    audio_dir: Optional[str] = None
    audio_file: Optional[str] = None
    dataset_manifest: Optional[str] = None
    audio_type: str = "wav"

    # Output and batching
    output_manifest: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    sort_by_duration: bool = True
    random_seed: Optional[int] = None

    # Unified streaming context in seconds. Converted to subsampled encoder frames.
    chunk_secs: float = 2.0
    left_context_secs: float = 10.0
    right_context_secs: float = 2.0
    att_context_size: Optional[list] = None
    att_context_size_as_chunk: bool = True

    # Device and precision. Cache-aware Conformer inference currently expects float32.
    cuda: Optional[int] = None
    allow_mps: bool = False
    amp: bool = False
    amp_dtype: str = "float16"
    compute_dtype: Optional[str] = "float32"
    matmul_precision: str = "high"

    # Decoding strategy for RNNT models.
    decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))

    # Prompt conditioning: target language for models with lang ID prompt (e.g. "en", "de").
    target_lang: Optional[str] = None

    # Cache warm-up strategy:
    #   "growing_cache" — incrementally grow attention cache from empty (v2, recommended)
    #   "dual_path"     — full-context + cache-aware dual forward during warm-up
    #   "none"          — start with zero-initialized full-size cache (original)
    cache_mode: str = "growing_cache"

    # Only used when cache_mode="dual_path": number of warm-up steps.
    # -1 = auto (ceil(last_channel_cache_size / valid_out_len)), 0 = disabled.
    warmup_steps: int = -1

    # WER calculation.
    calculate_wer: bool = True
    use_cer: bool = False
    debug_mode: bool = False
    calculate_rtfx: bool = False


def extract_transcriptions(hyps):
    """Extract text from RNNT hypotheses."""
    if isinstance(hyps[0], Hypothesis):
        return [hyp.text for hyp in hyps]
    return hyps


def resolve_prompt_id(asr_model, target_lang: Optional[str]) -> Optional[int]:
    """Resolve target language into the model's prompt id."""
    if not getattr(asr_model, "use_prompt", False):
        return None

    if target_lang is None:
        logging.warning(
            "Model supports prompt conditioning but target_lang is not set. "
            "Running without language prompt may produce suboptimal results."
        )
        return None

    prompt_dict = asr_model.prompt_dictionary
    prompt_id = prompt_dict.get(target_lang)
    if prompt_id is None:
        prompt_id = prompt_dict.get("unk")
    if prompt_id is None:
        available = list(prompt_dict.keys())[:10]
        raise ValueError(
            f"Unknown target_lang='{target_lang}' and no 'unk' fallback in prompt_dictionary. "
            f"Available: {available}{'...' if len(prompt_dict) > 10 else ''}"
        )

    logging.info(f"Prompt conditioning enabled: target_lang='{target_lang}' -> prompt_id={prompt_id}")
    return prompt_id


def build_prompt_tensor(asr_model, processed_signal: torch.Tensor, compute_dtype: torch.dtype, prompt_id: Optional[int]):
    """Build a one-hot prompt tensor aligned to the cached encoder chunk."""
    if prompt_id is None:
        return None

    hidden_length = math.ceil(processed_signal.shape[-1] / asr_model.encoder.subsampling_factor)
    prompt = torch.zeros(
        processed_signal.shape[0],
        hidden_length,
        asr_model.num_prompts,
        dtype=compute_dtype,
        device=processed_signal.device,
    )
    prompt[:, :, prompt_id] = 1.0
    return prompt


def configure_unified_streaming_context(asr_model, cfg: TranscriptionConfig):
    """Configure unified encoder [left, chunk, right] context and return timing metadata."""
    audio_sample_rate = asr_model.cfg.preprocessor.sample_rate
    feature_stride_sec = asr_model.cfg.preprocessor.window_stride
    features_per_sec = 1.0 / feature_stride_sec
    encoder_subsampling_factor = asr_model.encoder.subsampling_factor

    context_encoder_frames = ContextSize(
        left=int(cfg.left_context_secs * features_per_sec / encoder_subsampling_factor),
        chunk=int(cfg.chunk_secs * features_per_sec / encoder_subsampling_factor),
        right=int(cfg.right_context_secs * features_per_sec / encoder_subsampling_factor),
    )

    if cfg.att_context_size is not None:
        context_encoder_frames = ContextSize(
            left=int(cfg.att_context_size[0]),
            chunk=int(cfg.att_context_size[1]),
            right=int(cfg.att_context_size[2]),
        )

    if context_encoder_frames.chunk <= 0:
        raise ValueError("The chunk context must be at least one subsampled encoder frame.")

    if asr_model.cfg.encoder.att_context_style != "chunked_limited_with_rc":
        raise ValueError(
            "This script is intended for unified RNNT models with "
            "encoder.att_context_style='chunked_limited_with_rc'."
        )

    att_context_size = [
        context_encoder_frames.left,
        context_encoder_frames.chunk,
        context_encoder_frames.right,
    ]
    if cfg.att_context_size_as_chunk:
        asr_model.encoder.att_context_size = att_context_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=context_encoder_frames.chunk + context_encoder_frames.right,
            shift_size=context_encoder_frames.chunk,
            att_context_size=att_context_size,
        )
        # Align cache size to a chunk boundary so that the chunked_limited_with_rc
        # attention mask's chunk grid lines up with the cache/input boundary.
        # Without this, position cache_len falls mid-chunk, misaligning attention.
        chunk_frames = context_encoder_frames.chunk
        aligned_cache = math.ceil(asr_model.encoder.streaming_cfg.last_channel_cache_size / chunk_frames) * chunk_frames
        asr_model.encoder.streaming_cfg.last_channel_cache_size = aligned_cache
    else:
        asr_model.encoder.setup_streaming_params()

    # Fix conv cache_drop_size for symmetric (non-causal) convolutions.
    # CausalConv1D.update_cache() pads _right_padding zeros on the right before slicing
    # the cache, so cache_drop_size must also cover those zeros to keep
    # only chunk frames (not right-context frames) in the conv cache.
    for m in asr_model.encoder.layers.modules():
        if isinstance(m, CausalConv1D) and m._right_padding > 0:
            m.cache_drop_size = asr_model.encoder.streaming_cfg.cache_drop_size + m._right_padding

    features_frame2audio_samples = make_divisible_by(
        int(audio_sample_rate * feature_stride_sec),
        factor=encoder_subsampling_factor,
    )
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor
    context_samples = ContextSize(
        left=context_encoder_frames.left * encoder_frame2audio_samples,
        chunk=context_encoder_frames.chunk * encoder_frame2audio_samples,
        right=context_encoder_frames.right * encoder_frame2audio_samples,
    )
    latency_secs = (context_samples.chunk + context_samples.right) / audio_sample_rate

    logging.info(
        "Corrected contexts (sec): "
        f"Left {context_samples.left / audio_sample_rate:.2f}, "
        f"Chunk {context_samples.chunk / audio_sample_rate:.2f}, "
        f"Right {context_samples.right / audio_sample_rate:.2f}"
    )
    logging.info(f"Corrected contexts (subsampled encoder frames): {context_encoder_frames}")
    logging.info(f"Encoder streaming config: {asr_model.encoder.streaming_cfg}")
    logging.info(
        f"Attention cache aligned to {asr_model.encoder.streaming_cfg.last_channel_cache_size} frames "
        f"(multiple of chunk_size={context_encoder_frames.chunk})"
    )
    logging.info(f"Theoretical latency: {latency_secs:.2f} seconds")

    return context_encoder_frames, context_samples, features_frame2audio_samples, encoder_subsampling_factor


def perform_streaming(
    asr_model,
    audio_batch: torch.Tensor,
    audio_batch_lengths: torch.Tensor,
    context_samples: ContextSize,
    features_frame2audio_samples: int,
    encoder_subsampling_factor: int,
    compute_dtype: torch.dtype,
    debug_mode: bool = False,
    prompt_id: Optional[int] = None,
    cache_mode: str = "growing_cache",
    warmup_steps: int = 0,
):
    """Run cached encoder streaming and incremental RNNT decoding for one raw-audio batch.

    Supports three cache modes:

    * ``"growing_cache"`` — attention cache starts empty and grows incrementally.
      First chunk is processed in near-offline mode (single encoder pass, no waste).
    * ``"dual_path"``     — first ``warmup_steps`` chunks use a full-context forward
      for the decoder while a parallel ``cache_aware_stream_step`` populates caches.
    * ``"none"``          — original fixed-size zero-initialized cache.
    """
    batch_size = audio_batch.shape[0]
    device = audio_batch.device
    decoding_computer = asr_model.decoding.decoding.decoding_computer
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

    use_growing_cache = cache_mode == "growing_cache"

    if use_growing_cache:
        cache_last_channel = None
        cache_last_time = None
        cache_last_channel_len = None
    else:
        cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
            batch_size=batch_size,
            dtype=compute_dtype,
            device=device,
        )

    current_batched_hyps = None
    state = None
    left_sample = 0
    right_sample = min(context_samples.chunk + context_samples.right, audio_batch.shape[1])
    buffer = StreamingBatchedAudioBuffer(
        batch_size=batch_size,
        context_samples=context_samples,
        dtype=audio_batch.dtype,
        device=device,
    )
    rest_audio_lengths = audio_batch_lengths.clone()
    pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size
    if isinstance(pre_encode_cache_size, list):
        pre_encode_cache_size = pre_encode_cache_size[1]

    step_idx = 0

    while left_sample < audio_batch.shape[1]:
        chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample
        is_last_chunk_batch = chunk_length >= rest_audio_lengths
        is_last_chunk = right_sample >= audio_batch.shape[1]
        chunk_lengths_batch = torch.where(
            is_last_chunk_batch,
            rest_audio_lengths,
            torch.full_like(rest_audio_lengths, fill_value=chunk_length),
        )
        buffer.add_audio_batch_(
            audio_batch[:, left_sample:right_sample],
            audio_lengths=chunk_lengths_batch,
            is_last_chunk=is_last_chunk,
            is_last_chunk_batch=is_last_chunk_batch,
        )

        use_warmup = cache_mode == "dual_path" and warmup_steps > 0 and step_idx < warmup_steps
        is_v2_first_step = use_growing_cache and cache_last_channel is None

        processed_signal, processed_signal_length = asr_model.preprocessor(
            input_signal=buffer.samples,
            length=buffer.context_size_batch.total(),
        )
        feature_context = buffer.context_size.subsample(factor=features_frame2audio_samples)
        feature_context_batch = buffer.context_size_batch.subsample(factor=features_frame2audio_samples)

        if is_v2_first_step:
            # --- GROWING CACHE (v2): first step — full features, no trimming ---
            any_last = is_last_chunk_batch.any().item()
            prompt = build_prompt_tensor(
                asr_model=asr_model,
                processed_signal=processed_signal,
                compute_dtype=compute_dtype,
                prompt_id=prompt_id,
            )
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = asr_model.encoder.cache_aware_stream_step_v2(
                processed_signal=processed_signal.to(compute_dtype),
                processed_signal_length=processed_signal_length,
                cache_last_channel=None,
                cache_last_time=None,
                cache_last_channel_len=None,
                keep_all_outputs=any_last,
                drop_extra_pre_encoded=0,
            )
            if getattr(asr_model, "use_prompt", False) and prompt is not None:
                encoded = asr_model._apply_prompt(encoded, prompt)

            encoder_output = encoded.transpose(1, 2)
            encoder_context = buffer.context_size.subsample(factor=encoder_frame2audio_samples)
            encoder_context_batch = buffer.context_size_batch.subsample(factor=encoder_frame2audio_samples)
            encoder_output = encoder_output[:, encoder_context.left:]
            chunk_out_len = torch.where(
                is_last_chunk_batch,
                encoded_len - encoder_context_batch.left,
                encoder_context_batch.chunk,
            )

        elif use_warmup:
            # --- DUAL PATH: full-context forward for decoder, cache_aware for cache ---
            full_signal = processed_signal.to(compute_dtype)
            full_signal_len = processed_signal_length.clone()
            encoded_full, encoded_full_len = asr_model.encoder(
                audio_signal=full_signal, length=full_signal_len,
            )
            if prompt_id is not None and getattr(asr_model, "use_prompt", False):
                input_time = buffer.samples.shape[1]
                hidden_length = math.ceil(
                    input_time / (features_frame2audio_samples * encoder_subsampling_factor)
                )
                prompt_tensor = torch.zeros(
                    batch_size, hidden_length, asr_model.num_prompts,
                    dtype=compute_dtype, device=device,
                )
                prompt_tensor[:, :, prompt_id] = 1.0
                encoded_full = asr_model._apply_prompt(encoded_full, prompt_tensor)

            encoder_output = encoded_full.transpose(1, 2)
            encoder_context = buffer.context_size.subsample(factor=encoder_frame2audio_samples)
            encoder_context_batch = buffer.context_size_batch.subsample(factor=encoder_frame2audio_samples)
            encoder_output = encoder_output[:, encoder_context.left:]
            chunk_out_len = torch.where(
                is_last_chunk_batch,
                encoded_full_len - encoder_context_batch.left,
                encoder_context_batch.chunk,
            )

            left_feature_start = max(feature_context.left - pre_encode_cache_size, 0)
            processed_signal_cached = processed_signal[:, :, left_feature_start:].to(compute_dtype)
            processed_signal_length_cached = (processed_signal_length - left_feature_start).clamp(min=0)
            drop_extra_pre_encoded = (
                asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
                if feature_context.left > left_feature_start
                else 0
            )
            (
                _,
                _,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = asr_model.encoder.cache_aware_stream_step(
                processed_signal=processed_signal_cached,
                processed_signal_length=processed_signal_length_cached,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )

        else:
            # --- NORMAL / v2 subsequent steps: cache-aware path ---
            left_feature_start = max(feature_context.left - pre_encode_cache_size, 0)
            processed_signal_trimmed = processed_signal[:, :, left_feature_start:].to(compute_dtype)
            processed_signal_length_trimmed = (processed_signal_length - left_feature_start).clamp(min=0)
            drop_extra_pre_encoded = (
                asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
                if feature_context.left > left_feature_start
                else 0
            )
            prompt = build_prompt_tensor(
                asr_model=asr_model,
                processed_signal=processed_signal_trimmed,
                compute_dtype=compute_dtype,
                prompt_id=prompt_id,
            )
            any_last = is_last_chunk_batch.any().item()

            stream_step_fn = (
                asr_model.encoder.cache_aware_stream_step_v2 if use_growing_cache
                else asr_model.encoder.cache_aware_stream_step
            )
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = stream_step_fn(
                processed_signal=processed_signal_trimmed,
                processed_signal_length=processed_signal_length_trimmed,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=any_last,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            if getattr(asr_model, "use_prompt", False) and prompt is not None:
                encoded = asr_model._apply_prompt(encoded, prompt)

            encoder_output = encoded.transpose(1, 2)
            chunk_out_len = torch.where(
                is_last_chunk_batch,
                encoded_len,
                feature_context_batch.chunk // encoder_subsampling_factor,
            )

        chunk_batched_hyps, _, state = decoding_computer(
            x=encoder_output,
            out_len=chunk_out_len,
            prev_batched_state=state,
            multi_biasing_ids=None,
        )

        if current_batched_hyps is None:
            current_batched_hyps = chunk_batched_hyps
        else:
            current_batched_hyps.merge_(chunk_batched_hyps)

        if debug_mode:
            cache_size = cache_last_channel.size(2) if cache_last_channel is not None else 0
            logging.info(
                f"  step={step_idx} left_sample={left_sample} | "
                f"mode={'v2_first' if is_v2_first_step else ('warmup' if use_warmup else cache_mode)} | "
                f"cache_size={cache_size} | out_len={chunk_out_len.tolist()} | "
                f"is_last={is_last_chunk_batch.tolist()} | "
                f"new_tokens={chunk_batched_hyps.current_lengths.tolist()}"
            )
            hyps = batched_hyps_to_hypotheses(current_batched_hyps, None, batch_size=batch_size)
            transcribed_texts = [asr_model.tokenizer.ids_to_text(hyp.y_sequence.tolist()) for hyp in hyps]
            logging.info(f"  cumulative texts: {extract_transcriptions(transcribed_texts)}")

        rest_audio_lengths -= chunk_lengths_batch
        left_sample = right_sample
        right_sample = min(right_sample + context_samples.chunk, audio_batch.shape[1])
        step_idx += 1

    if current_batched_hyps is None:
        return ["" for _ in range(batch_size)]

    final_hyps = batched_hyps_to_hypotheses(current_batched_hyps, None, batch_size=batch_size)
    final_streaming_tran = [asr_model.tokenizer.ids_to_text(hyp.y_sequence.tolist()) for hyp in final_hyps]
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")
    return final_streaming_tran



def prepare_records(cfg: TranscriptionConfig):
    """Load input records from a manifest, directory, or single audio file."""
    if sum((cfg.audio_file is not None, cfg.dataset_manifest is not None, cfg.audio_dir is not None)) != 1:
        raise ValueError("Exactly one of `audio_file`, `dataset_manifest`, or `audio_dir` should be provided.")

    if cfg.audio_file is not None:
        records = [{"audio_filepath": cfg.audio_file}]
        dataset_title = Path(cfg.audio_file).stem
    elif cfg.dataset_manifest is not None:
        records = read_manifest(cfg.dataset_manifest)
        manifest_dir = Path(cfg.dataset_manifest).parent.absolute()
        for record in records:
            record["audio_filepath"] = str(filepath_to_absolute(record["audio_filepath"], manifest_dir))
        dataset_title = Path(cfg.dataset_manifest).stem
    else:
        assert cfg.audio_dir is not None
        filepaths = glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True)
        records = [{"audio_filepath": audio_filepath} for audio_filepath in filepaths]
        dataset_title = Path(cfg.audio_dir).name

    for idx, record in enumerate(records):
        record["_input_order"] = idx
        if "duration" not in record:
            record["duration"] = librosa.get_duration(path=record["audio_filepath"])

    if cfg.sort_by_duration:
        records.sort(key=lambda item: item["duration"], reverse=True)

    return records, dataset_title


def write_predictions(records, predictions, output_manifest: str, pred_text_attr_name: str = "pred_text", use_cer=False):
    """Write JSONL predictions and return references if present."""
    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)

    refs = []
    hyps = []
    with open(output_manifest, "w", encoding="utf-8") as out_f:
        for record, pred_text in zip(records, predictions):
            output_record = {k: v for k, v in record.items() if k != "_input_order"}
            output_record[pred_text_attr_name] = pred_text
            if "text" in output_record:
                refs.append(output_record["text"])
                hyps.append(pred_text)
                output_record["wer"] = round(
                    word_error_rate(hypotheses=[pred_text], references=[output_record["text"]], use_cer=use_cer)
                    * 100,
                    2,
                )
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    return hyps, refs


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    cfg = OmegaConf.structured(cfg)
    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None.")

    device = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)
    if (cfg.compute_dtype is not None and cfg.compute_dtype != "float32") and cfg.amp:
        raise ValueError("amp=true is mutually exclusive with a compute_dtype other than float32.")

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    compute_dtype = torch.float32 if cfg.amp else get_inference_dtype(cfg.compute_dtype, device=device)
    if compute_dtype != torch.float32:
        raise NotImplementedError("Cache-aware Conformer inference currently supports only float32 compute dtype.")

    asr_model, model_name = setup_model(cfg=cfg, map_location=device)
    if not isinstance(asr_model, (EncDecRNNTModel, EncDecHybridRNNTCTCModel)):
        raise ValueError("This script supports RNNT and hybrid RNNT/CTC models only.")

    prompt_id = resolve_prompt_id(asr_model=asr_model, target_lang=cfg.target_lang)

    with open_dict(cfg.decoding):
        cfg.decoding.fused_batch_size = -1
        cfg.decoding.beam.return_best_hypothesis = True

    if isinstance(asr_model, EncDecRNNTModel):
        asr_model.change_decoding_strategy(cfg.decoding)
    if hasattr(asr_model, "cur_decoder"):
        asr_model.change_decoding_strategy(cfg.decoding, decoder_type="rnnt")

    asr_model = asr_model.to(device=device, dtype=compute_dtype)
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.freeze()
    asr_model.eval()

    (
        _context_encoder_frames,
        context_samples,
        features_frame2audio_samples,
        encoder_subsampling_factor,
    ) = configure_unified_streaming_context(asr_model=asr_model, cfg=cfg)

    cache_mode = cfg.cache_mode
    if cache_mode == "dual_path":
        if cfg.warmup_steps == -1:
            valid_out_len = asr_model.encoder.streaming_cfg.valid_out_len
            cache_size = asr_model.encoder.streaming_cfg.last_channel_cache_size
            warmup_steps = math.ceil(cache_size / valid_out_len) if valid_out_len > 0 else 0
        else:
            warmup_steps = cfg.warmup_steps
    else:
        warmup_steps = 0
    logging.info(f"Cache mode: {cache_mode} | warmup_steps: {warmup_steps}")

    records, dataset_title = prepare_records(cfg)
    if cfg.output_manifest is None:
        cfg.output_manifest = f"{model_name}_{dataset_title}_unified_cache_aware_streaming.json"

    audio_dataset = SimpleAudioDataset(
        audio_filenames=[record["audio_filepath"] for record in records],
        sample_rate=asr_model.cfg.preprocessor.sample_rate,
    )
    audio_dataloader = DataLoader(
        dataset=audio_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=AudioBatch.collate_fn,
        drop_last=False,
        in_order=True,
    )

    all_predictions = []
    total_audio_duration = sum(float(record.get("duration", 0.0)) for record in records)
    start_time = time.time()
    record_offset = 0

    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=cfg.amp), torch.no_grad(), torch.inference_mode():
        for audio_data in audio_dataloader:
            batch_size = audio_data.audio_signals.shape[0]
            logging.info(
                f"Starting cache-aware streaming for samples {record_offset} to {record_offset + batch_size - 1}."
            )
            predictions = perform_streaming(
                asr_model=asr_model,
                audio_batch=audio_data.audio_signals.to(device=device),
                audio_batch_lengths=audio_data.audio_signal_lengths.to(device=device),
                context_samples=context_samples,
                features_frame2audio_samples=features_frame2audio_samples,
                encoder_subsampling_factor=encoder_subsampling_factor,
                compute_dtype=compute_dtype,
                debug_mode=cfg.debug_mode,
                prompt_id=prompt_id,
                cache_mode=cache_mode,
                warmup_steps=warmup_steps,
            )
            all_predictions.extend(predictions)
            record_offset += batch_size

    elapsed = time.time() - start_time
    if cfg.sort_by_duration:
        restored = sorted(
            zip(records, all_predictions),
            key=lambda records_predictions: records_predictions[0]["_input_order"],
        )
        records, all_predictions = map(list, zip(*restored))

    hyps, refs = write_predictions(records, all_predictions, cfg.output_manifest, use_cer=cfg.use_cer)
    logging.info(f"Finished writing predictions to {cfg.output_manifest}.")

    if cfg.calculate_wer and len(refs) == len(hyps) and len(refs) > 0:
        wer = word_error_rate(hypotheses=hyps, references=refs, use_cer=cfg.use_cer)
        metric_name = "CER" if cfg.use_cer else "WER"
        logging.info(f"{metric_name}% of cache-aware streaming mode: {round(wer * 100, 2)}")

    logging.info(f"The whole cache-aware streaming inference process took: {elapsed:.2f}s")
    if cfg.calculate_rtfx and elapsed > 0.0:
        logging.info(f"RTFx: {total_audio_duration / elapsed:.2f}")

    return cfg


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
