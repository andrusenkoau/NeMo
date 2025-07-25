
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from typing import Optional
from omegaconf import DictConfig
from nemo.collections.asr.parts.mixins.mixins import lens_to_mask
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.utils import logging


@dataclass
class AEDStreamingState:
    decoder_input_ids: torch.Tensor = None  # tokens ids of initial canary prompt
    tgt: torch.Tensor = None  # buffer with deocoded tokens ids
    decoding_step: int = -1  # current decoding step
    decoder_mems_list: list = None  # decoder caches, helps to reduce the memory usage
    is_last_chunk_batch: torch.Tensor = False  # whether the current chunk is the last speech chunk in the audio
    max_generation_length: int = 512  # maximum number of tokens to be generated for each sample
    max_tokens_per_alignatt_step: int = (
        10  # maximum number of tokens to be generated for each step of alignatt decoding policy (before the last speech chunk)
    )
    device: torch.device = None


class GreedyBatchedStreamingAEDComputer(ABC):
    """
    Batched streaming AED decoding with support for waitk and alignatt decoding policies.
    """

    def __init__(
        self,
        asr_model,
        frame_chunk_size,
        decoding_cfg,
        debug_mode=False,
    ):
        """
        Init method.
        Args:

        """
        super().__init__()

        self.asr_model = asr_model
        self.frame_chunk_size = frame_chunk_size
        self.decoding_cfg = decoding_cfg
        self.state = AEDStreamingState()
        self.debug_mode = debug_mode

    def __call__(self, encoder_output, encoder_output_len, prev_batched_state):

        self.state = prev_batched_state

        # prepare encoder embeddings for the decoding
        # enc_states = encoder_output.permute(0, 2, 1)
        encoded_speech = self.asr_model.encoder_decoder_proj(encoder_output)

        encoder_input_mask = lens_to_mask(encoder_output_len, encoded_speech.shape[1]).to(
            encoded_speech.dtype
        )

        # initiall waitk lagging. Applicable for waitk and alignatt decoding policies
        if encoded_speech.size(-2) // self.frame_chunk_size < self.decoding_cfg.waitk_lagging and torch.any(
            torch.logical_not(self.state.is_last_chunk_batch)
        ):
            # need to wait for more speech
            if self.debug_mode:
                logging.warning(f"!!! need more initial speech according to the waitk policy !!!")
                logging.warning(f"[encoded_speech.shape]: {encoded_speech.shape}")

        # wait-k decoding policy
        if self.decoding_cfg.streaming_policy == "waitk":
            if self.state.decoding_step < 0:
                # first decoding step
                tgt, batch_size, _ = self.asr_model.decoding.decoding.greedy_search._prepare_for_search(
                    self.state.decoder_input_ids,
                    encoded_speech,
                )
                input_ids = tgt
            else:
                input_ids = self.state.tgt[
                    self.state.batch_idxs, self.state.current_context_lengths - 1
                ].unsqueeze(-1)

            active_samples_inner_loop = (
                torch.ones(self.state.batch_size, dtype=torch.bool, device=self.state.device) * self.state.active_samples
            )
            decoder_mems_list = self.state.decoder_mems_list

            # define start and max generation lengths
            start_from = self.state.decoding_step + 1
            if torch.any(torch.logical_not(self.state.is_last_chunk_batch)):
                # predict only one token per speech chunk if not the last one
                max_generation_length = start_from + 1
            else:
                max_generation_length = self.decoding_cfg.max_generation_length

            # inner deocding loop (with same speech chunk)
            for i in range(start_from, max_generation_length):

                if not decoder_mems_list:
                    positional_indexes = torch.zeros_like(self.state.current_context_lengths)
                else:
                    positional_indexes = self.state.current_context_lengths - 1

                logits, decoder_mems_list, xatt_scores_list = (
                    self.asr_model.decoding.decoding.greedy_search._one_step_forward(
                        input_ids,
                        encoded_speech,
                        encoder_input_mask,
                        decoder_mems_list,
                        positional_indexes,
                        return_scores=False,
                        return_xatt_scores=True,
                    )
                )
                next_tokens = torch.argmax(logits[:, -1], dim=-1)
                text_tokens = self.asr_model.tokenizer.ids_to_tokens(next_tokens.tolist())

                # compute eos tokens mask
                is_eos_tokens = next_tokens == self.asr_model.tokenizer.eos
                # rearange active samples (inner loop) depends on eos prediction
                active_samples_inner_loop *= torch.logical_not(is_eos_tokens)
                # disable samples (upper loop) with eos and end of speech
                eos_and_end_speech_mask = is_eos_tokens * self.state.is_last_chunk_batch
                self.state.active_samples = self.state.active_samples * torch.logical_not(
                    eos_and_end_speech_mask
                )

                if self.debug_mode:
                    logging.warning(f"-------------" * 5)
                    logging.warning(f"decoding step (i)        : {i}")
                    logging.warning(f"start_from               : {start_from}")
                    logging.warning(f"max_generation_length    : {max_generation_length}")
                    logging.warning(f"[encoded_speech.shape]   : {encoded_speech.shape}")
                    logging.warning(f"[is_last_chunk_batch]   : {self.state.is_last_chunk_batch}")
                    logging.warning(f"[active_samples]         : {self.state.active_samples}")
                    logging.warning(f"[current_context_lengths]: {self.state.current_context_lengths}")
                    logging.warning(f"[predicted token]        : {text_tokens}")
                    logging.warning(f"[predicted token id]     : {next_tokens}")

                if not torch.any(active_samples_inner_loop):
                    if self.debug_mode:
                        logging.warning(f"!#! no active samples in inner loop, do next upper step !#!")
                    break

                # write predicted tokens to the tgt tensor
                torch.where(
                    active_samples_inner_loop, next_tokens, self.state.eos_tokens, out=next_tokens
                )
                self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths] = next_tokens

                # canary_data.decoding_step = i
                self.state.decoding_step += input_ids.size(-1)
                # input_ids = next_tokens.unsqueeze(-1)
                # input_ids = canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths].unsqueeze(-1)

                # check for hallucinations
                # TODO add more consequtive tokens? Now we are checking only 3 same tokens
                hallucination_mask = torch.logical_and(
                    self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths]
                    == self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths - 1],
                    self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths - 1]
                    == self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths - 2],
                )
                if torch.any(hallucination_mask):
                    logging.warning(f"!!! hallucination detected !!!")
                    self.state.active_samples *= torch.logical_not(hallucination_mask)
                    active_samples_inner_loop *= torch.logical_not(hallucination_mask)

                self.state.current_context_lengths += active_samples_inner_loop
                input_ids = self.state.tgt[
                    self.state.batch_idxs, self.state.current_context_lengths - 1
                ].unsqueeze(-1)

                # disable samples with maximum context length
                samples_with_max_context_length = (
                    self.state.current_context_lengths == self.decoding_cfg.max_generation_length - 1
                )
                if torch.any(samples_with_max_context_length * self.state.active_samples):
                    logging.warning(f"!!! maximum context length reached !!!")
                    self.state.active_samples *= torch.logical_not(samples_with_max_context_length)
                    active_samples_inner_loop *= torch.logical_not(samples_with_max_context_length)

                # zero out decoder_mems_list for non active samples
                if torch.any(torch.logical_not(active_samples_inner_loop)):
                    for j in range(len(decoder_mems_list)):
                        decoder_mems_list[j][:, -1] *= active_samples_inner_loop.unsqueeze(-1)
                self.state.decoder_mems_list = decoder_mems_list

                if self.debug_mode:
                    # import ipdb; ipdb.set_trace()
                    pass

        # alignatt decoding policy
        elif self.decoding_cfg.streaming_policy == "alignatt":
            if canary_data.decoding_step < 0:
                # first decoding step
                tgt, batch_size, _ = self.decoding.decoding.greedy_search._prepare_for_search(
                    canary_data.decoder_input_ids,
                    canary_data.encoded_speech,
                )
                input_ids = tgt
                start_from = 0
            else:
                input_ids = canary_data.tgt[
                    canary_data.batch_idxs, canary_data.current_context_lengths - 1
                ].unsqueeze(-1)
                start_from = torch.min(canary_data.current_context_lengths).item() - 1

            decoder_mems_list = canary_data.decoder_mems_list
            canary_data.steps_per_inner_loop = torch.zeros(batch_size, dtype=torch.long, device=self.state.device)
            canary_data.active_samples_inner_loop = (
                torch.ones(batch_size, dtype=torch.bool, device=self.state.device) * canary_data.active_samples
            )

            for i in range(start_from, canary_data.max_generation_length):
                # prepare positional indexes offset for attention decoder
                if not decoder_mems_list:
                    positional_indexes = torch.zeros_like(canary_data.current_context_lengths)
                else:
                    positional_indexes = canary_data.current_context_lengths - 1

                logits, decoder_mems_list, xatt_scores_list = (
                    self.decoding.decoding.greedy_search._one_step_forward(
                        input_ids,
                        canary_data.encoded_speech,
                        encoder_input_mask,
                        decoder_mems_list,
                        positional_indexes,
                        return_scores=False,
                        return_xatt_scores=True,
                    )
                )

                next_tokens = torch.argmax(logits[:, -1], dim=-1)
                text_token = self.tokenizer.ids_to_tokens(next_tokens.tolist())

                # compute the most attended encoder token
                xatt_scores = xatt_scores_list[cfg.xatt_scores_layer]
                xatt_scores = torch.mean(xatt_scores, 1)
                if i == 0 and xatt_scores.shape[-1] <= cfg.exclude_sink_frames:
                    exclude_sink_frames = xatt_scores.shape[-1] - 2
                else:
                    exclude_sink_frames = cfg.exclude_sink_frames
                most_attended_idxs = (
                    torch.argmax(xatt_scores[:, :, exclude_sink_frames:], dim=-1) + exclude_sink_frames
                )

                if cfg.use_avgpool_for_alignatt:
                    average_pooling_xatt_scores = canary_data.avgpool2d(xatt_scores[:, :, exclude_sink_frames:])
                    most_attended_idxs_avgpool = (
                        torch.argmax(average_pooling_xatt_scores, dim=-1) + exclude_sink_frames
                    )
                    most_attended_idxs = most_attended_idxs_avgpool

                # select the last attended token for each sample
                if most_attended_idxs.size(-1) > 1:
                    most_attended_idxs = most_attended_idxs[:, -1]
                else:
                    most_attended_idxs = most_attended_idxs.squeeze(-1)

                # aligatt condition (True -- continue decoding, False -- wait for more speech)
                alignatt_condition = (
                    canary_data.encoded_speech.shape[1] - (most_attended_idxs + 1) >= cfg.alignatt_thr
                )

                # alignatt condition is always True for the last speech chunk
                alignatt_condition += canary_data.is_last_chunk_batch

                # applay alignatt condition for inner loop
                canary_data.active_samples_inner_loop *= alignatt_condition

                if cfg.debug_mode:
                    logging.warning(f"-------------" * 5)
                    logging.warning(f"canary_data.decoding_step  : {canary_data.decoding_step}")
                    logging.warning(f"decoding step i: {i}")
                    logging.warning(f"[encoded_speech.shape]     : {canary_data.encoded_speech.shape}")
                    logging.warning(f"[positional_indexes]     : {positional_indexes}")
                    logging.warning(f"[most_attended_idxs]       : {most_attended_idxs}")
                    logging.warning(f"[is_last_chunk_batch]     : {canary_data.is_last_chunk_batch}")
                    logging.warning(f"[active_samples]           : {canary_data.active_samples}")
                    logging.warning(f"[active_samples_inner_loop]: {canary_data.active_samples_inner_loop}")
                    logging.warning(f"[current_context_lengths]  : {canary_data.current_context_lengths}")
                    logging.warning(f"[predicted tokens]         : {text_token}")
                    logging.warning(f"[predicted tokens id]: {next_tokens}")
                    import pdb

                    pdb.set_trace()

                # increase speech chunk if no active samples in the inner loop
                if not torch.any(canary_data.active_samples_inner_loop):
                    if cfg.debug_mode:
                        logging.warning(f"!#! no active samples in inner loop, do next upper step !#!")
                    break

                # compute eos tokens mask
                # TODO add a case of "." + EOS prediction. It is the important case for AST tasl with PC support
                is_eos_tokens = next_tokens == self.tokenizer.eos
                # rearange active samples (inner loop) depends on eos prediction
                canary_data.active_samples_inner_loop *= torch.logical_not(is_eos_tokens)
                # disable samples (upper loop) with eos and end of speech
                eos_and_end_speech_mask = is_eos_tokens * canary_data.is_last_chunk_batch
                canary_data.active_samples *= torch.logical_not(eos_and_end_speech_mask)

                if not torch.any(canary_data.active_samples_inner_loop):
                    if cfg.debug_mode:
                        logging.warning(f"!#! no active samples in inner loop, do next upper step !#!")
                    break

                # write predicted tokens to the tgt tensor
                torch.where(
                    canary_data.active_samples_inner_loop, next_tokens, canary_data.eos_tokens, out=next_tokens
                )
                canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths] = next_tokens

                # update tokens frame alignment based on current encoder step (this alignment is used for LAAL calculation)
                canary_data.tokens_frame_alignment[canary_data.batch_idxs, canary_data.current_context_lengths] = (
                    canary_data.encoded_speech.size(-2)
                )

                canary_data.decoding_step += input_ids.size(-1)
                # input_ids = next_tokens.unsqueeze(-1)

                # check for hallucinations
                # TODO add more consequtive tokens? Now we are checking only 3 same tokens
                hallucination_mask = torch.logical_and(
                    canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths]
                    == canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths - 1],
                    canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths - 1]
                    == canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths - 2],
                )
                if torch.any(hallucination_mask):
                    logging.warning(f"!!! hallucination detected !!!")
                    canary_data.active_samples *= torch.logical_not(hallucination_mask)
                    canary_data.active_samples_inner_loop *= torch.logical_not(hallucination_mask)

                # disable samples with maximum context length
                samples_with_max_context_length = (
                    canary_data.current_context_lengths == canary_data.max_generation_length - 1
                )
                if torch.any(samples_with_max_context_length * canary_data.active_samples):
                    logging.warning(f"!!! maximum context length reached !!!")
                    canary_data.active_samples *= torch.logical_not(samples_with_max_context_length)
                    canary_data.active_samples_inner_loop *= torch.logical_not(samples_with_max_context_length)

                # zero out decoder_mems_list for non active samples
                # TODO it does not work if first token was EOS
                if torch.any(torch.logical_not(canary_data.active_samples_inner_loop)):
                    for j in range(len(decoder_mems_list)):
                        decoder_mems_list[j][:, -1] *= canary_data.active_samples_inner_loop.unsqueeze(-1)

                canary_data.decoder_mems_list = decoder_mems_list
                canary_data.current_context_lengths += canary_data.active_samples_inner_loop
                # TODO model does not predicts any real tokens in the case of first EOS prediction
                input_ids = canary_data.tgt[
                    canary_data.batch_idxs, canary_data.current_context_lengths - 1
                ].unsqueeze(-1)

                # limit number of steps per inner loop if not end of speech
                canary_data.steps_per_inner_loop += canary_data.active_samples_inner_loop
                disable_samples_mask = canary_data.steps_per_inner_loop == canary_data.max_tokens_per_alignatt_step
                disable_samples_mask *= torch.logical_not(canary_data.is_last_chunk_batch)
                canary_data.active_samples_inner_loop *= torch.logical_not(disable_samples_mask)

                if cfg.debug_mode:
                    logging.warning(f"-------------" * 5)
                    logging.warning(f"canary_data.decoding_step  : {canary_data.decoding_step}")
                    logging.warning(f"decoding step i: {i}")
                    logging.warning(f"[encoded_speech.shape]     : {canary_data.encoded_speech.shape}")
                    logging.warning(f"[positional_indexes]     : {positional_indexes}")
                    logging.warning(f"[most_attended_idxs]       : {most_attended_idxs}")
                    logging.warning(f"[is_last_chunk_batch]     : {canary_data.is_last_chunk_batch}")
                    logging.warning(f"[active_samples]           : {canary_data.active_samples}")
                    logging.warning(f"[active_samples_inner_loop]: {canary_data.active_samples_inner_loop}")
                    logging.warning(f"[current_context_lengths]  : {canary_data.current_context_lengths}")
                    logging.warning(f"[predicted tokens]         : {text_token}")
                    logging.warning(f"[predicted tokens id]: {next_tokens}")

                if cfg.debug_mode:
                    import pdb

                    pdb.set_trace()

        else:
            raise ValueError("Canary streaming decoding supports only alignatt or waitk decodong policy")
        
        return None, None, self.state

