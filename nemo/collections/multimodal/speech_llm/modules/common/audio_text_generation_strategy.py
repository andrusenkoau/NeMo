# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional, Tuple

import torch

import nemo.collections.nlp.modules.common.text_generation_strategy as text_generation_strategy
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import shift_tokens_by_multi_audios
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids

from nemo.utils import logging
import numpy as np

# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    boolean = boolean.unsqueeze(0).unsqueeze(-1)
    return (1 - boolean) * val1 + boolean * val2


class AudioToTextGenerationStrategy(text_generation_strategy.GPTModelTextGenerationStrategy):
    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """initialize the batch data before the inference steps."""
        # Move to GPU.

        audio_feats, audio_feat_lens = self.model.perception(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        if num_audios is not None:
            # handle multiple audio files per sample
            audio_feats = audio_feats.split(num_audios.tolist())
            audio_feat_lens = audio_feat_lens.split(num_audios.tolist())

        encoder_input, attention_mask, _, position_ids, encoder_max_length = self.model.inject_perception_input(
            audio_feats, audio_feat_lens, context_tokens, context_lengths, context_start_idx
        )

        self.attention_mask = attention_mask
        self.position_ids = position_ids

        if num_audios is not None:
            # handle multiple audio files per sample
            new_context_tokens = shift_tokens_by_multi_audios(
                context_tokens, context_lengths, audio_feat_lens, context_start_idx, encoder_max_length
            )
            audio_feat_lens = torch.stack([torch.sum(lens) for lens in audio_feat_lens])  # [batch,]
        else:
            new_context_tokens = self.model._shift_labels_by_emb_len(
                context_tokens, context_lengths, audio_feat_lens, encoder_max_length, pad_token=0
            )

        return new_context_tokens, encoder_input, audio_feat_lens

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""
        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use)
            started = context_lengths <= curr_context_length
            embeddings2use = switch(input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use, started)

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        else:
            tokenizer = self.model.tokenizer
            conditions = []
            end_tokens = set()
            end_tokens.add(eod_id)
            for end_string in end_strings:
                if len(end_string) > 1:
                    continue
                ids_1 = tokenizer.text_to_ids(f'<extra_id_1>{end_string}')
                ids_2 = tokenizer.text_to_ids('<extra_id_1>')
                if len(ids_1) <= len(ids_2):
                    continue
                token_id = ids_1[len(ids_2) :][0]

                end_tokens.add(token_id)

            for p, token_item in zip(prev, tokens):
                text = tokenizer.ids_to_text(token_item.tolist())
                conditions.append(
                    any([text.endswith(end_string) for end_string in end_strings] + [p.item() in end_tokens])
                )
            return torch.tensor(conditions, dtype=torch.bool, device=tokens.device)


class CrossAttendAudioToTextGenerationStrategy(AudioToTextGenerationStrategy):
    
    def read_next_audio_chunk(self, **strategy_args):
        
        if strategy_args["debug_mode"]:
            logging.warning("\n" + "*******"*10)
            logging.warning(f"read_next_audio_chunk")
            logging.warning("*******"*10 + "\n")

        audio_length = self.audio_length
        audio_signal = self.audio_signal[:]
        
        # read params
        pre_decision_ratio = strategy_args['pre_decision_ratio']
        sample_rate = strategy_args.get('sample_rate', 16000)
        right_context = strategy_args.get('right_context', 13)
        audio_encoder_fs = strategy_args.get('audio_encoder_fs', 80)

        # read audio chunk
        cur_enc_len = pre_decision_ratio * (self.speech_chunk_step + strategy_args['alignatt_waitk'])
        self.speech_chunk_step += 1
        cur_src_len = (cur_enc_len + right_context) * audio_encoder_fs * sample_rate // 1000
        audio_signal = audio_signal[:, :cur_src_len]
        audio_length = torch.minimum(
            audio_length, torch.from_numpy(np.array([cur_src_len])).to(audio_length.device)
        )
        if strategy_args["debug_mode"]:
            logging.warning(f"cur_enc_len: {cur_enc_len}")

        # [b, t, c] run audio encoder model
        speech_encoded, speech_encoded_len = self.model.perception(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            processed_signal=None,
            processed_signal_length=None,
        )
        # truncate to the current encoder length
        speech_encoded = speech_encoded[:, :cur_enc_len]
        speech_encoded_len = torch.from_numpy(np.array([cur_enc_len])).to(speech_encoded_len.device)

        self.cur_speech_encoded = speech_encoded
        self.cur_speech_encoded_len = speech_encoded_len

        self.audio_signal_is_finished = False
        cur_enc_len_in_samples = cur_enc_len * audio_encoder_fs * sample_rate // 1000
        self.part_of_processed_audio = cur_enc_len_in_samples / self.audio_length
        # logging.warning(f"cur_enc_len_in_samples: {cur_enc_len_in_samples}")
        # logging.warning(f"audio_length: {self.audio_length}")
        # logging.warning(f"self.part_of_processed_audio: {self.part_of_processed_audio}")
        if  cur_enc_len_in_samples >= audio_length:
            logging.warning(f"audio_signal_is_finished: {cur_enc_len}")
            self.audio_signal_is_finished = True

        return speech_encoded, speech_encoded_len, cur_enc_len

    
    def init_batch_per_step(
        self,
        step: int,
        **strategy_args,
    ):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        context_lengths = self.context_lengths
        cl = context_lengths[0]
        assert torch.equal(context_lengths, torch.ones_like(context_lengths) * cl)
        audio_length = self.audio_length
        audio_signal = self.audio_signal[:]

        # logging.warning(f"strategy_args['decode_policy']: {strategy_args['decode_policy']}")
        if  strategy_args['decode_policy'] == 'alignatt':
            # increase speech chunk size
            speech_encoded, speech_encoded_len, cur_enc_len = self.read_next_audio_chunk(**strategy_args)

            # call xattn for step 0
            input_embeds = self.model._get_text_embeddings(self.context_tokens, None).transpose(0, 1)
            if step == 0:
                assert torch.equal(context_lengths, torch.ones_like(context_lengths) * context_lengths[0])
                context_length = context_lengths[0]
                # empty fixed feature for attention masking in context tokens
                encoder_input_prev, self.extra_outputs = self.model.perception_cross_attn(
                    torch.zeros_like(speech_encoded[:, :1]),
                    torch.ones_like(speech_encoded_len),
                    input_embeds[:, : context_length - 1],
                    input_lengths=context_lengths - 1,
                    return_mems=True,
                )

                # compute xatt of speech and current context tokens
                apply_xatt = True
                max_steps_with_same_token = 10
                step_count = 0
                decoder_mems_list = self.extra_outputs.get('decoder_mems_list', None)
                exclude_sink_frames = strategy_args.get('exclude_sink_frames', 2)
                exclude_sink_frames = exclude_sink_frames // 2
                while apply_xatt and step_count < max_steps_with_same_token:
                    encoder_input, self.extra_outputs = self.model.perception_cross_attn(
                        speech_encoded,
                        speech_encoded_len,
                        input_embeds[:, context_length - 1 : context_length],
                        input_lengths=torch.ones_like(context_lengths),
                        return_mems=True,
                        decoder_mems_list=decoder_mems_list,
                    )
                    step_count += 1
                    # compute the most attended audio frame for each text token from the specified cross attention layer
                    # [batch, head, text_context, audio]
                    xatt_scores = self.extra_outputs['xatt_scores_list'][strategy_args['xatt_layer']]
                    xatt_scores = torch.mean(xatt_scores, 1)
                    # exclude first audio tokens (sink tokens)
                    # most_attended_idx = torch.argmax(xatt_scores[:,:,2:], dim=-1)+2
                    # alternative serach for the most attended frame by smoothing the attention scores
                    most_attended_idx = torch.argmax(xatt_scores[:,:,exclude_sink_frames:], dim=-1)+exclude_sink_frames
                    average_pooling_xatt_scores = self.pool2d(xatt_scores[:,:,exclude_sink_frames:])
                    most_attended_idx_pool = torch.argmax(average_pooling_xatt_scores, dim=-1)+exclude_sink_frames
                    if strategy_args["debug_mode"]:
                        logging.warning(f"=== most_attended_idx: {most_attended_idx[0].item()}")
                        logging.warning(f"=== most_attended_idx_pool: {most_attended_idx_pool[0].item()}")  
                        logging.warning(f"with value: {xatt_scores[:,:, most_attended_idx].item():.4f}")
                        logging.warning(f"xatt_scores: {xatt_scores[-1,:,-8:]}")
                    if cur_enc_len-1 - most_attended_idx >= strategy_args['alignatt_thr']:
                        apply_xatt = False
                    else:
                        if strategy_args["debug_mode"]:
                            logging.warning(f"[zero step]: !!! condition is not OK, increase speech chunk size")
                        # increase speech chunk size
                        speech_encoded, speech_encoded_len, cur_enc_len = self.read_next_audio_chunk(**strategy_args)

                # combain all the embeddings
                encoder_input = torch.cat([encoder_input_prev, encoder_input, input_embeds[:, context_length:]], dim=1)
                # logging.warning(f"encoder_input.shape: {encoder_input.shape}")
                base_module = self.model.model
                lm_embedding = (
                    base_module.language_model.embedding
                    if hasattr(base_module, 'language_model')
                    else base_module.embedding
                )
                self.attention_mask = self.model._create_attention_mask(encoder_input)
                if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
                    encoder_input = encoder_input.transpose(0, 1).contiguous()
            else:
                encoder_input = input_embeds.transpose(0, 1).contiguous()
                self.attention_mask = self.model._create_attention_mask(encoder_input)
            context_tokens = self.context_tokens

        elif strategy_args['decode_policy'] == 'waitk':
            waitk_lagging = strategy_args['waitk_lagging']
            pre_decision_ratio = strategy_args['pre_decision_ratio']
            sample_rate = strategy_args.get('sample_rate', 16000)
            right_context = strategy_args.get('right_context', 13)
            audio_encoder_fs = strategy_args.get('audio_encoder_fs', 80)
            # for now only support sharing the same text context for a batch
            cur_enc_len = pre_decision_ratio * (step + waitk_lagging)
            cur_src_len = (cur_enc_len + right_context) * audio_encoder_fs * sample_rate // 1000
            
            # logging.warning(f"cur_enc_len: {cur_enc_len}")
            # logging.warning(f"self.context_tokens.shape: {self.context_tokens.shape}")
            # logging.warning(f"self.context_tokens: {self.context_tokens}")
            # logging.warning(f"context_lengths: {context_lengths}")
            # raise NotImplementedError("This function is not implemented yet")
            
            audio_signal = audio_signal[:, :cur_src_len]
            audio_length = torch.minimum(
                audio_length, torch.from_numpy(np.array([cur_src_len])).to(audio_length.device)
            )

            # [b, t, c]
            speech_encoded, speech_encoded_len = self.model.perception(
                input_signal=audio_signal,
                input_signal_length=audio_length,
                processed_signal=None,
                processed_signal_length=None,
            )
            # call xattn for step 0
            input_embeds = self.model._get_text_embeddings(self.context_tokens, None).transpose(0, 1)
            if step == 0:
                assert torch.equal(context_lengths, torch.ones_like(context_lengths) * context_lengths[0])
                context_length = context_lengths[0]
                # empty fixed feature for attention masking in context tokens
                encoder_input_prev, self.extra_outputs = self.model.perception_cross_attn(
                    torch.zeros_like(speech_encoded[:, :1]),
                    torch.ones_like(speech_encoded_len),
                    input_embeds[:, : context_length - 1],
                    input_lengths=context_lengths - 1,
                    return_mems=True,
                )
                # the first answer token prediction
                decoder_mems_list = self.extra_outputs.get('decoder_mems_list', None)
                encoder_input, self.extra_outputs = self.model.perception_cross_attn(
                    speech_encoded,
                    speech_encoded_len,
                    input_embeds[:, context_length - 1 : context_length],
                    input_lengths=torch.ones_like(context_lengths),
                    return_mems=True,
                    decoder_mems_list=decoder_mems_list,
                )

                encoder_input = torch.cat([encoder_input_prev, encoder_input, input_embeds[:, context_length:]], dim=1)
                base_module = self.model.model
                lm_embedding = (
                    base_module.language_model.embedding
                    if hasattr(base_module, 'language_model')
                    else base_module.embedding
                )
                self.attention_mask = self.model._create_attention_mask(encoder_input)
                if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
                    encoder_input = encoder_input.transpose(0, 1).contiguous()
            else:
                encoder_input = input_embeds.transpose(0, 1).contiguous()
                self.attention_mask = self.model._create_attention_mask(encoder_input)
            context_tokens = self.context_tokens
        else:
            # raise NotImplementedError("This function is not implemented yet")
            batch = {
                'audio_signal': audio_signal,
                'audio_signal_length': audio_length,
                'tokens': self.context_tokens,
                'contexts': self.context_tokens,
                'tokens_length': context_lengths,
                'context_lengths': context_lengths,  # used by waitk
                'labels': self.context_tokens,
                'loss_mask': None,
            }
            (
                encoder_input,
                self.attention_mask,
                context_tokens,
                _,
                (speech_encoded, speech_encoded_len, extra_outputs),
            ) = self.model.prepare_llm_input(batch, **strategy_args)

        if strategy_args['decode_policy'] in ['alignatt', 'waitk']:
            # get rid of right context
            speech_encoded_len = torch.minimum(
                speech_encoded_len, torch.from_numpy(np.array([cur_enc_len])).to(speech_encoded_len.device)
            )
            speech_encoded = speech_encoded[:, :cur_enc_len]
            self.cur_speech_encoded = speech_encoded
            self.cur_speech_encoded_len = speech_encoded_len
        self.position_ids = build_position_ids(encoder_input[:, :, 0].transpose(0, 1))
        return (
            context_tokens,
            (encoder_input, speech_encoded, speech_encoded_len),
            torch.zeros_like(context_lengths),
        )

    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
        **strategy_args,
    ):
        self.audio_signal = audio_signal[:]
        self.audio_length = audio_length[:]
        self.context_tokens = context_tokens[:]
        self.context_lengths = context_lengths[:]

        return self.init_batch_per_step(0, **strategy_args)

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
        force_increase_speech_chunk: bool = False,
        **strategy_args,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        input_embeddings, speech_encoded, speech_encoded_len = input_embeddings

        # logging.warning(f"*********"*10)
        # logging.warning(f"step: {step}")
        # raise StopIteration


        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
            # logging.warning(f"embeddings2use.shape: {embeddings2use.shape}")
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            # take last predicted token
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1)
            # logging.warning(f"tokens2use: {tokens2use}")
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use).transpose(0, 1)
            started = context_lengths <= curr_context_length
            # for seq started, first get embeddings2use, and then run cross attend, after that replace embeddings2use with the cross attended embed
            # use speech_encoded; rerun cross attend
            # [1, b, d]
            decoder_mems_list = self.extra_outputs.get('decoder_mems_list', None)
            if decoder_mems_list is not None:
                decoder_mems_list = decoder_mems_list[:, :, : curr_context_length - 1]
            if  strategy_args['decode_policy'] == 'waitk':
                # for now only support sharing the same text context for a batch
                assert torch.equal(context_lengths, torch.ones_like(context_lengths) * context_lengths[0])
                (_, (_, cur_speech_encoded, cur_speech_encoded_len), _) = self.init_batch_per_step(
                    step, **strategy_args
                )
            elif strategy_args['decode_policy'] == 'alignatt':
                # check alighatt condition by xatt before increasing speech chunk
                
                # 0. force to increase speech chunk size -- in the case of eos prediction before the end of the audio signal
                if force_increase_speech_chunk:
                    if strategy_args["debug_mode"]:
                        logging.warning(f"eos prediction at {self.part_of_processed_audio.item():.2f} part of the audio signal, force to increase speech chunk size")
                    cur_speech_encoded, cur_speech_encoded_len, _ = self.read_next_audio_chunk(**strategy_args)
                    self.cur_speech_encoded = cur_speech_encoded
                    self.cur_speech_encoded_len = cur_speech_encoded_len

                # 1. compute xatt of speech and current context tokens
                apply_xatt = True
                max_steps_with_same_token = 10 # max number of speech increasing steps with the same token
                step_count = 0
                decoder_mems_list = self.extra_outputs.get('decoder_mems_list', None)
                exclude_sink_frames = strategy_args.get('exclude_sink_frames', 2)

                while apply_xatt and step_count < max_steps_with_same_token:
                    embeddings2use_tmp, self.extra_outputs_tmp = self.model.perception_cross_attn(
                        self.cur_speech_encoded,
                        self.cur_speech_encoded_len,
                        embeddings2use,
                        input_lengths=tokens2use.squeeze(-1) != self.model.tokenizer.eos_id,
                        decoder_mems_list=decoder_mems_list,
                        return_mems=True,
                    )
                    # 2. compute the most attended audio frame for each text token from the second cross attention layer
                    # logging.warning(f"self.extra_outputs['xatt_scores_list'][0].shape: {self.extra_outputs['xatt_scores_list'][0].shape}")
                    # logging.warning(f"self.extra_outputs['xatt_scores_list'][1].shape: {self.extra_outputs['xatt_scores_list'][1].shape}")
                    step_count += 1
                    xatt_scores = self.extra_outputs_tmp['xatt_scores_list'][strategy_args['xatt_layer']]
                    xatt_scores = torch.mean(xatt_scores, 1) # mean by attention heads
                    # most_attended_idx = torch.argmax(xatt_scores, dim=-1)
                    most_attended_idx = torch.argmax(xatt_scores[:,:,exclude_sink_frames:], dim=-1)+exclude_sink_frames
                    sum_of_last_xatt = torch.sum(xatt_scores[:,:, -strategy_args['alignatt_thr']:])

                    # alternative serach for the most attended frame by smoothing the attention scores
                    average_pooling_xatt_scores = self.pool2d(xatt_scores[:,:,exclude_sink_frames:])
                    most_attended_idx_pool = torch.argmax(average_pooling_xatt_scores, dim=-1)+exclude_sink_frames

                    if strategy_args["debug_mode"]:
                        logging.warning(f"self.cur_speech_encoded_len: {self.cur_speech_encoded_len.item()}")
                        logging.warning(f"=== most_attended_idx: {most_attended_idx[0].item()}")
                        logging.warning(f"=== most_attended_idx_pool: {most_attended_idx_pool[0].item()}") 
                        logging.warning(f"with value: {xatt_scores[:,:, most_attended_idx].item():.4f}")
                        logging.warning(f"xatt_scores: {xatt_scores[-1,:,-8:]}")
                        logging.warning(f"+++sum_of_last_xatt: {sum_of_last_xatt}")
                    # # alignatt policy
                    if self.audio_signal_is_finished or \
                        self.cur_speech_encoded_len-1 - most_attended_idx_pool >= strategy_args['alignatt_thr'] or \
                            step_count == max_steps_with_same_token-1:
                    # EDatt policy
                    # if self.audio_signal_is_finished or \
                    #     sum_of_last_xatt < 0.1 or \
                    #         step_count == max_steps_with_same_token-1:
                        # condition is OK, just use the same speech chunk
                        apply_xatt = False
                        if strategy_args["debug_mode"]:
                            logging.warning(f"condition is OK, just use the same speech chunk")
                        embeddings2use = embeddings2use_tmp
                        self.extra_outputs = self.extra_outputs_tmp
                        """Prepare batch for each of the inference steps"""
                        setkey_value_array = torch.tensor(
                            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
                        )
                        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

                        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
                        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
                        return batch, tensor_shape
                    else:
                        if strategy_args["debug_mode"]:
                            logging.warning(f"!!! condition is not OK, increase speech chunk size")
                        # for now only support sharing the same text context for a batch
                        assert torch.equal(context_lengths, torch.ones_like(context_lengths) * context_lengths[0])
                        
                        cur_speech_encoded, cur_speech_encoded_len, _ = self.read_next_audio_chunk(**strategy_args)
                        self.cur_speech_encoded = cur_speech_encoded
                        self.cur_speech_encoded_len = cur_speech_encoded_len

            else:
                cur_speech_encoded = speech_encoded
                cur_speech_encoded_len = speech_encoded_len

            # need to use audio_ratio field if to support text-only decoding
            embeddings2use, self.extra_outputs = self.model.perception_cross_attn(
                cur_speech_encoded,
                cur_speech_encoded_len,
                embeddings2use,
                input_lengths=tokens2use.squeeze(-1) != self.model.tokenizer.eos_id,
                decoder_mems_list=decoder_mems_list,
                return_mems=True,
            )

            embeddings2use = switch(
                input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use.transpose(0, 1), started
            )

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.multimodal.speech_llm.models.modular_models import (
        CrossAttendModularAudioGPTModel,
        ModularAudioGPTModel,
    )

    if isinstance(model, CrossAttendModularAudioGPTModel):
        return CrossAttendAudioToTextGenerationStrategy(model, **args)
    elif isinstance(model, ModularAudioGPTModel):
        return AudioToTextGenerationStrategy(model, **args)
    else:
        return text_generation_strategy.model_inference_strategy_dispatcher(model, **args)
