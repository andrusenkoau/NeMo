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

import torch
import triton
import triton.language as tl


@triton.jit
def _rnnt_joint_fwd_kernel(
    encoder_output_ptr,
    predictor_output_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_scores_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T log probs. Stores result in `target_scores_out_ptr` and `blank_scores_out_ptr`.
    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    # load lengths for source/target
    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i >= source_len or target_i > target_len:
        # no calculations required
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    ...


@triton.jit
def _rnnt_joint_bwd_kernel(
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for Joint RNN-T log probs.
    We recalculate part of the forward here to avoid using extra memory in forward.
    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    ...


class RnntJointLogProbs(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T, supporting torch.autograd.
    """

    @staticmethod
    def forward(
        ctx,
        encoder_output_projected: torch.Tensor,
        predictor_output_projected: torch.Tensor,
        targets: torch.Tensor,
        tgt_lengths: torch.Tensor,
        src_lengths: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        blank_id: int,
        activation: str = "relu",
        dropout_p: float = 0.0,
    ):
        """

        Args:
            ctx: ctx object for storing the context
            blank_id: id of the blank output
            src_lengths: tensor with lengths for source utterances
            tgt_lengths: tensor with lengths for targets

        Returns:

        """
        if activation != "relu":
            # TODO: support other activations later, no need to implement now
            raise NotImplementedError

        if dropout_p != 0.0:
            # TODO: support dropout later, no need to implement now
            raise NotImplementedError

        # Use float64 if input is float64, otherwise float32
        use_fp64 = encoder_output_projected.dtype == torch.float64

        encoder_output_projected = encoder_output_projected.contiguous()
        predictor_output_projected = predictor_output_projected.contiguous()
        targets = targets.contiguous()

        device = encoder_output_projected.device
        batch_size, src_max_length = encoder_output_projected.shape[:-1]
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = bias.shape[0]
        target_logprobs = encoder_output_projected.new_zeros([batch_size, src_max_length, tgt_max_length_plus_1])
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.zeros_like(blank_logprobs)

        float_dtype = torch.float64 if use_fp64 else torch.float32

        # run Triton kernel
        _rnnt_joint_fwd_kernel[(batch_size, src_max_length, tgt_max_length_plus_1)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            # max_source_len=logits.shape[1],
            # max_target_len_plus_1=logits.shape[2],
            # num_labels=logits.shape[3],
            # blank_id=blank_id,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_scores_out_ptr=log_sum_exp_scores,
            BLOCK_SIZE=triton.next_power_of_2(vocab_size),
            USE_FP64=use_fp64,
        )

        # saving for backward
        # ctx.save_for_backward(logits, targets, src_lengths, tgt_lengths, log_sum_exp_scores)
        # ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        """
        Backward calculation for RNN-T log-probs.

        Args:
            ctx: ctx object for storing the context
            grad_target_scores: upstream gradient for target scores
            grad_blank_scores:  upstream gradient for blank scores

        Returns:
            gradient for encoder_output and predictor_output, None for all other arguments for `forward`
        """
        # (logits, targets, src_lengths, tgt_lengths, log_sum_exp_scores) = ctx.saved_tensors
        # blank_id = ctx.blank_id
        # use_fp64 = ctx.use_fp64
        # grad_logits = torch.zeros_like(logits)
        # _rnnt_logprobs_bwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
        #     logits_ptr=logits,
        #     grad_logits_out_ptr=grad_logits,
        #     src_lengths_ptr=src_lengths,
        #     tgt_lengths_ptr=tgt_lengths,
        #     targets_ptr=targets,
        #     log_sum_exp_scores_ptr=log_sum_exp_scores,
        #     max_source_len=logits.shape[1],
        #     max_target_len_plus_1=logits.shape[2],
        #     num_labels=logits.shape[3],
        #     blank_id=blank_id,
        #     grad_target_scores_ptr=grad_target_scores,
        #     grad_blank_scores_ptr=grad_blank_scores,
        #     BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
        #     USE_FP64=use_fp64,
        # )
        return None, None, None, None, None, None, None, None, None


def rnnt_joint_logprobs_triton(
    encoder_output_projected: torch.Tensor,
    predictor_output_projected: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    activation: str = "relu",
    dropout_p: float = 0.0,
):

    target_logprobs, blank_logprobs = RnntJointLogProbs.apply(
        encoder_output_projected,
        predictor_output_projected,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        activation,
        dropout_p,
    )
    return target_logprobs, blank_logprobs
