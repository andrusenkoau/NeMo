# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#

"""
# This script would evaluate an N-gram language model trained with KenLM library (https://github.com/kpu/kenlm) in
# fusion with beam search decoders on top of a trained ASR Transducer model. NeMo's beam search decoders are capable of using the
# KenLM's N-gram models to find the best candidates. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# You may train the LM model with 'scripts/ngram_lm/train_kenlm.py'.

# Config Help

To discover all arguments of the script, please run :
python eval_beamsearch_ngram.py --help
python eval_beamsearch_ngram.py --cfg job

# USAGE

python eval_beamsearch_ngram_transducer.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file \
           kenlm_model_file=<path to the binary KenLM model> \
           beam_width=[<list of the beam widths, separated with commas>] \
           beam_alpha=[<list of the beam alphas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null \
           decoding_strategy=<greedy_batch or maes decoding>
           maes_prefix_alpha=[<list of the maes prefix alphas, separated with commas>] \
           maes_expansion_gamma=[<list of the maes expansion gammas, separated with commas>] \
           hat_subtract_ilm=<in case of HAT model: subtract internal LM or not> \
           hat_ilm_weight=[<in case of HAT model: list of the HAT internal LM weights, separated with commas>] \
           ...


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

           beam_width=[4,8,16,....] \
           beam_alpha=[-2.0,-1.0,...,1.0,2.0] \

# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

"""


import contextlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional

import editdistance
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.collections.asr.parts.k2.context_graph import ContextGraph
from nemo.collections.asr.parts.k2.context_graph_ctc import ContextGraphCTC
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

from word_boosting_search import recognize_wb

from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceConstants,
    ConfidenceMethodConfig,
    ConfidenceMethodConstants,
)

# fmt: off


@dataclass
class EvalWordBoostingConfig:
    """
    Evaluate an ASR model with beam search decoding and n-gram KenLM language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    probs_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    acoustic_batch_size: int = 128  # The batch size to calculate log probabilities
    beam_batch_size: int = 128  # The batch size to be used for beam search decoding
    device: str = "cuda"  # The device to load the model onto to calculate log probabilities
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities
    num_workers: int = 1  # Number of workers for DataLoader
    
    # for hybrid model
    decoder_type: Optional[str] = None # [ctc, rnnt] Decoder type for hybrid ctc-rnnt model

    # The decoding scheme to be used for evaluation
    decoding_strategy: str = "greedy_batch" # ["greedy_batch", "beam", "tsd", "alsd", "maes"]


    ### Context Biasing ###:
    applay_context_biasing: bool = True
    context_score: float = 4.0  # per token weight for context biasing words
    context_file: Optional[str] = None  # string with context biasing words (words splitted by space)

    sort_logits: bool = True # do logits sorting before decoding - it reduces computation on puddings
    softmax_temperature: float = 1.00
    preserve_alignments: bool = True

    decoding: rnnt_beam_decoding.BeamRNNTInferConfig = rnnt_beam_decoding.BeamRNNTInferConfig(beam_size=128)



# fmt: on


def merge_alignment_with_wb_hyps(
    candidate,
    model,
    wb_result,
):
    
    alignment = candidate.alignments

    alignment_per_frame = []
    for idx, items in enumerate(alignment):
        for item in items:
            token = item[1].item()
            if token != model.decoder.blank_idx:
                alignment_per_frame.append([idx, model.tokenizer.ids_to_tokens([token])[0]])
    alignment_tokens = alignment_per_frame

    # alignment_per_frame = []
    # for items in alignment:
    #     current_frame_ali = [x[1].item() for x in items]
    #     # logging.warning("-----"*10)
    #     # logging.warning(current_frame_ali)
    #     alignment_per_frame.append(current_frame_ali)

    # # get words borders
    # alignment_tokens = []
    # for idx, frame_ali in enumerate(alignment_per_frame):
    #     for idy, token in enumerate(frame_ali):
    #         if token != model.decoder.blank_idx:
    #             alignment_tokens.append([idx, model.tokenizer.ids_to_tokens([token])[0]])

    if not alignment_tokens:
        for wb_hyp in wb_result:
            print(f"wb_hyp: {wb_hyp.word}")
        return " ".join([wb_hyp.word for wb_hyp in wb_result])


    slash = "▁"
    word_alignment = []
    word = ""
    l, r, = None, None
    for item in alignment_tokens:
        if not word:
            word = item[1][1:]
            l = item[0]
            r = item[0]
        else:
            if item[1].startswith(slash):
                word_alignment.append((word, l, r))
                word = item[1][1:]
                l = item[0]
                r = item[0]
            else:
                word += item[1]
                r = item[0]
    word_alignment.append((word, l, r))
    ref_text = [item[0] for item in word_alignment]
    ref_text = " ".join(ref_text)
    print(f"rnnt_word_alignment: {word_alignment}")

    # merge wb_hyps and word alignment:

    for wb_hyp in wb_result:
        new_word_alignment = []
        already_inserted = False
        # lh, rh = wb_hyp.start_frame, wb_hyp.end_frame
        wb_interval = set(range(wb_hyp.start_frame, wb_hyp.end_frame+1))
        for item in word_alignment:
            li, ri = item[1], item[2]
            item_interval = set(range(item[1], item[2]+1))
            if wb_hyp.start_frame < li:
                if not already_inserted:
                    new_word_alignment.append((wb_hyp.word, wb_hyp.start_frame, wb_hyp.end_frame))
                    already_inserted = True

            intersection_part = 100/len(item_interval) * len(wb_interval & item_interval)
            if intersection_part < 30:
                new_word_alignment.append(item)
            elif not already_inserted:
                new_word_alignment.append((wb_hyp.word, wb_hyp.start_frame, wb_hyp.end_frame))
                already_inserted = True
        # insert last wb word:
        if not already_inserted:
            new_word_alignment.append((wb_hyp.word, wb_hyp.start_frame, wb_hyp.end_frame))

        word_alignment = new_word_alignment
        print(f"wb_hyp: {wb_hyp.word:<10} -- ({wb_hyp.start_frame}, {wb_hyp.end_frame})")

    boosted_text_list = [item[0] for item in new_word_alignment]
    boosted_text = " ".join(boosted_text_list)
    print(f"before: {ref_text}")
    print(f"after : {boosted_text}")
    
    return boosted_text


def decoding_step(
    model: nemo_asr.models.ASRModel,
    cfg: EvalWordBoostingConfig,
    all_probs: List[torch.Tensor],
    target_transcripts: List[str],
    audio_file_paths: List[str],
    durations: List[str],
    preds_output_file: str = None,
    preds_output_manifest: str = None,
    beam_batch_size: int = 128,
    progress_bar: bool = True,
    wb_results: List[dict] = None,
):
    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)
    # Reset config
    model.change_decoding_strategy(None)

    # cfg.decoding.hat_ilm_weight = cfg.decoding.hat_ilm_weight * cfg.hat_subtract_ilm
    # Override the beam search config with current search candidate configuration
    cfg.decoding.return_best_hypothesis = False
    # cfg.decoding.ngram_lm_model = cfg.kenlm_model_file
    # cfg.decoding.hat_subtract_ilm = cfg.hat_subtract_ilm

    # preserve aligmnet:
    model.cfg.decoding.preserve_alignments = cfg.preserve_alignments

    # Update model's decoding strategy config
    model.cfg.decoding.strategy = cfg.decoding_strategy
    model.cfg.decoding.beam = cfg.decoding

    # Update model's decoding strategy
    model.change_decoding_strategy(model.cfg.decoding)
    logging.setLevel(level)

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    sample_idx = 0

    if preds_output_file:
        out_file = open(preds_output_file, 'w', encoding='utf_8', newline='\n')
    if preds_output_manifest:
        out_manifest = open(preds_output_manifest, 'w', encoding='utf_8', newline='\n')

    if progress_bar:
        description = "Greedy_batch decoding.."
        it = tqdm(range(int(np.ceil(len(all_probs) / beam_batch_size))), desc=description, ncols=120)
    else:
        it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
    for batch_idx in it:
        # disabling type checking
        probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
        probs_lens = torch.tensor([prob.shape[-1] for prob in probs_batch])
        with torch.no_grad():
            packed_batch = torch.zeros(len(probs_batch), probs_batch[0].shape[0], max(probs_lens), device='cpu')

            for prob_index in range(len(probs_batch)):
                packed_batch[prob_index, :, : probs_lens[prob_index]] = torch.tensor(
                    probs_batch[prob_index].unsqueeze(0), device=packed_batch.device, dtype=packed_batch.dtype
                )
            best_hyp_batch, beams_batch = model.decoding.rnnt_decoder_predictions_tensor(
                packed_batch, probs_lens, return_hypotheses=True,
            )
        beams_batch = [[x] for x in best_hyp_batch]

        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            target_split_c = list(target)
            words_count += len(target_split_w)
            chars_count += len(target_split_c)
            wer_dist_min = cer_dist_min = 10000
            audio_id = os.path.basename(audio_file_paths[sample_idx + beams_idx])
            for candidate_idx, candidate in enumerate(beams):  # type: (int, rnnt_beam_decoding.rnnt_utils.Hypothesis)
                
                ###################################
                if cfg.applay_context_biasing and wb_results[audio_file_paths[sample_idx + beams_idx]]:
                    
                    # make new text by mearging alignment with ctc-wb predictions:
                    print("----")
                    boosted_text = merge_alignment_with_wb_hyps(
                        candidate,
                        model,
                        wb_results[audio_file_paths[sample_idx + beams_idx]]
                    )
                    pred_text = boosted_text
                    beams[0].text = pred_text
                    print(f"ref   : {target}")
                    print("\n" + audio_file_paths[sample_idx + beams_idx])
                else:
                    pred_text = candidate.text
                
                #######################################

                # pred_text = candidate.text
                pred_split_w = pred_text.split()
                wer_dist = editdistance.eval(target_split_w, pred_split_w)
                pred_split_c = list(pred_text)
                cer_dist = editdistance.eval(target_split_c, pred_split_c)

                wer_dist_min = min(wer_dist_min, wer_dist)
                cer_dist_min = min(cer_dist_min, cer_dist)

                if candidate_idx == 0:
                    # first candidate
                    wer_dist_tosave = wer_dist
                    wer_dist_first += wer_dist
                    cer_dist_first += cer_dist

                score = candidate.score
                if preds_output_file:
                    
                    out_file.write('{}\t{}\t{}\n'.format(audio_id, pred_text, score))

                    #out_file.write('{}\t{}\n'.format(pred_text, score))
            wer_dist_best += wer_dist_min
            cer_dist_best += cer_dist_min

            # write manifest with prediction results
            alignment = []
            # if cfg.preserve_alignments:
            #     prev_frame_idx = None
            #     for i, items in enumerate(candidate.alignments):
            #         # token_id = item[0][1].item()
            #         token_id = [x[1].item() for x in items]
            #         # frame_idx = item[0][2]
            #         frame_idx = [x[2] for x in items]
            #         # if token_id == model.decoder.blank_idx:
            #         #     token_text = "-"
            #         # else:
            #         #     token_text = model.tokenizer.ids_to_tokens([token_id])[0]
            #         # alignment.append(f"{i}: {token_text}")
            #         alignment.append(f"{token_id}: {frame_idx}")

            if preds_output_manifest:
                item = {'audio_filepath': audio_file_paths[sample_idx + beams_idx],
                        'duration': durations[sample_idx + beams_idx],
                        'text': target_transcripts[sample_idx + beams_idx],
                        'pred_text': beams[0].text,
                        'wer': f"{wer_dist_tosave/len(target_split_w):.3f}",
                        'alignment': f"{alignment}"}
                out_manifest.write(json.dumps(item) + "\n")
        
        sample_idx += len(probs_batch)

    if cfg.decoding_strategy.startswith("greedy"):
        return wer_dist_first / words_count, cer_dist_first / chars_count

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of {cfg.decoding_strategy} decoding at '{preds_output_file}'.")

    logging.info(
            f"WER/CER with {cfg.decoding_strategy} decoding = {wer_dist_first / words_count:.2%}/{cer_dist_first / chars_count:.2%}"
        )
    logging.info(
        f"Oracle WER/CER in candidates with perfect LM= {wer_dist_best / words_count:.2%}/{cer_dist_best / chars_count:.2%}"
    )
    logging.info(f"=================================================================================")

    return wer_dist_first / words_count, cer_dist_first / chars_count


@hydra_runner(config_path=None, config_name='EvalWordBoostingConfig', schema=EvalWordBoostingConfig)
def main(cfg: EvalWordBoostingConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)  # type: EvalWordBoostingConfig

    valid_decoding_strategis = ["greedy", "greedy_batch"]
    if cfg.decoding_strategy not in valid_decoding_strategis:
        raise ValueError(
            f"Given decoding_strategy={cfg.decoding_strategy} is invalid. Available options are :\n"
            f"{valid_decoding_strategis}"
        )

    if cfg.nemo_model_file.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(cfg.nemo_model_file, map_location=torch.device(cfg.device))
        # asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[128, 128])
    else:
        logging.warning(
            "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.nemo_model_file, map_location=torch.device(cfg.device)
        )


    target_transcripts = []
    durations = []
    manifest_dir = Path(cfg.input_manifest).parent
    with open(cfg.input_manifest, 'r', encoding='utf_8') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {cfg.input_manifest} ...", ncols=120):
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            target_transcripts.append(data['text'])
            durations.append(data['duration'])
            audio_file_paths.append(str(audio_file.absolute()))

    if cfg.probs_cache_file and os.path.exists(cfg.probs_cache_file):
        logging.info(f"Found a pickle file of probabilities at '{cfg.probs_cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{cfg.probs_cache_file}' ...")
        with open(cfg.probs_cache_file, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{cfg.probs_cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:

        @contextlib.contextmanager
        def default_autocast():
            yield

        if cfg.use_amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast

            else:
                autocast = default_autocast
        else:

            autocast = default_autocast

        # manual calculation of encoder_embeddings
        with autocast():
            with torch.no_grad():
                asr_model.eval()
                asr_model.encoder.freeze()
                device = next(asr_model.parameters()).device
                all_probs = []
                ctc_logprobs = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                        for audio_file in audio_file_paths:
                            entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                            fp.write(json.dumps(entry) + '\n')
                    config = {
                        'paths2audio_files': audio_file_paths,
                        'batch_size': cfg.acoustic_batch_size,
                        'temp_dir': tmpdir,
                        'num_workers': cfg.num_workers,
                        'channel_selector': None,
                        'augmentor': None,
                    }
                    temporary_datalayer = asr_model._setup_transcribe_dataloader(config)

                    for test_batch in tqdm(temporary_datalayer, desc="Getting encoder and CTC decoder outputs...", disable=False):
                        encoded, encoded_len = asr_model.forward(
                            input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                        )
                        ctc_dec_outputs = asr_model.ctc_decoder(encoder_output=encoded).cpu()
                        # dump encoder embeddings per file
                        for idx in range(encoded.shape[0]):
                            encoded_no_pad = encoded[idx, :, : encoded_len[idx]]
                            ctc_dec_outputs_no_pad = ctc_dec_outputs[idx, : encoded_len[idx]]
                            all_probs.append(encoded_no_pad)
                            ctc_logprobs.append(ctc_dec_outputs_no_pad)

        if cfg.probs_cache_file:
            logging.info(f"Writing pickle files of probabilities at '{cfg.probs_cache_file}'...")
            with open(cfg.probs_cache_file, 'wb') as f_dump:
                pickle.dump(all_probs, f_dump)

################################_WB_PART_#########################

    wb_results = {}

    if cfg.applay_context_biasing:
        # load context graph:

        # ## bpe dropout:
        # kwl_set = set()
        # context_transcripts = []
        # sow_symbol_ids = asr_model.tokenizer.tokens_to_ids(['▁'])[0]
        # sow_symbol = '▁'
        # for line in open(cfg.context_file).readlines():
        #     word = line.strip().lower()
        #     tokenization = asr_model.tokenizer.tokenizer.encode(word) # , out_type=str
        #     # tokenization_tokens = asr_model.tokenizer.ids_to_tokens(tokenization)
        #     # tokenization_tokens = [token.replace(sow_symbol,"") for token in tokenization_tokens]
        #     # new_word = " ".join(tokenization_tokens)
        #     # tokenization = asr_model.tokenizer.tokenizer.encode(new_word) # , out_type=str
        #     kwl_set.add(str(tokenization))
        #     context_transcripts.append([tokenization, asr_model.tokenizer.text_to_ids(word)])
        #     # print(f"[BPE]: {word} -- {new_word}")
            
        #     for _ in range(50):
        #         tokenization = asr_model.tokenizer.tokenizer.encode(word, enable_sampling=True, alpha=0.1, nbest_size=-1)
        #         tokenization_tokens = asr_model.tokenizer.ids_to_tokens(tokenization)
        #         tokenization_tokens = [token.replace(sow_symbol,"") for token in tokenization_tokens]
        #         new_word = " ".join(tokenization_tokens)
        #         tokenization = asr_model.tokenizer.tokenizer.encode(new_word) # , out_type=str
        #         if tokenization[0] != sow_symbol_ids:
        #             tokenization_str = str(tokenization)
        #             if tokenization_str not in kwl_set:
        #                 kwl_set.add(tokenization_str)
        #                 context_transcripts.append([tokenization, asr_model.tokenizer.text_to_ids(word)])
        #                 print(f"[BPE dropout]: {word} -- {new_word}")


        # ## no bpe dropout:
        # context_transcripts = []
        # for line in open(cfg.context_file).readlines():
        #     word = line.strip().lower()
        #     context_transcripts.append([asr_model.tokenizer.text_to_ids(word),
        #                                 asr_model.tokenizer.text_to_ids(word)])

        ## no bpe dropout:
        context_transcripts = []
        for line in open(cfg.context_file).readlines():
            item = line.strip().lower().split("-")
            word = item[0]
            word_tokenization = [asr_model.tokenizer.text_to_ids(x) for x in item[1:]]
            context_transcripts.append([word, word_tokenization])



        context_graph = ContextGraphCTC(blank_id=asr_model.decoder.blank_idx)
        # logging.warning(context_transcripts)
        context_graph.build(context_transcripts)

        # run CTC based WB search:
        for idx, logits in tqdm(enumerate(ctc_logprobs), desc=f"CTC based word boosting...", ncols=120, total=len(ctc_logprobs)):
            # try:
            wb_result = recognize_wb(
                logits.numpy(),
                context_graph,
                asr_model,
                beam_threshold=5,        # 5
                context_score=5,         # 5 (4)
                keyword_thr=-5,          # -5
                ctc_ali_token_weight=3 # 3.0 (4.0)
            )
            # except:
            #     logging.warning("-------------------------")
            #     logging.warning(f"audio file is: {audio_file_paths[idx]}")
            wb_results[audio_file_paths[idx]] = wb_result
            print(f"ref: {target_transcripts[idx]}")
            print(audio_file_paths[idx] + "\n")
        


################################_WB_PART_#########################

    # sort all_probs according to length:
    if cfg.sort_logits:
        all_probs_with_indeces = (sorted(enumerate(all_probs), key=lambda x: x[1].size()[1], reverse=True))
        all_probs_sorted = []
        target_transcripts_sorted = []
        audio_file_paths_sorted = []
        durations_sorted = []
        for pair in all_probs_with_indeces:
            all_probs_sorted.append(pair[1])
            target_transcripts_sorted.append(target_transcripts[pair[0]])
            audio_file_paths_sorted.append(audio_file_paths[pair[0]])
            durations_sorted.append(durations[pair[0]])
        all_probs = all_probs_sorted
        target_transcripts = target_transcripts_sorted
        audio_file_paths = audio_file_paths_sorted
        durations = durations_sorted


    asr_model = asr_model.to('cpu')
    preds_output_file = os.path.join(cfg.preds_output_folder, f"recognition_results.tsv")
    preds_output_manifest = os.path.join(cfg.preds_output_folder, f"recognition_results.json")
    candidate_wer, candidate_cer = decoding_step(
        asr_model,
        cfg,
        all_probs=all_probs,
        target_transcripts=target_transcripts,
        audio_file_paths=audio_file_paths,
        durations=durations,
        beam_batch_size=cfg.beam_batch_size,
        progress_bar=True,
        preds_output_file=preds_output_file,
        preds_output_manifest=preds_output_manifest,
        wb_results=wb_results,
    )
    logging.info(f"Greedy batch WER/CER = {candidate_wer:.2%}/{candidate_cer:.2%}")


if __name__ == '__main__':
    main()
