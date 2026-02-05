# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
def _kl_div_fwd_kernel(
    teacher_logits_ptr,
    student_logits_ptr,
    mask_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    kl_loss_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T log probs. Stores result in `target_scores_ptr` and `blank_scores_ptr`.
    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    idx_no_vocab = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    mask_value = tl.load(mask_ptr + idx_no_vocab)
    if not mask_value:
        # no calculations required
        return

    # do all calculation at least in float32
    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # calculate offset in [B, T, U+1, V] tensor for the current vector with target logits
    idx_vocab_start = idx_no_vocab * num_labels
    teacher_logits_ptr += idx_vocab_start
    student_logits_ptr += idx_vocab_start
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    teacher_logits = tl.load(teacher_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    student_logits = tl.load(student_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    # stable log softmax calculation
    teacher_logits_max = tl.max(teacher_logits, axis=0)
    teacher_logits_minus_max = teacher_logits - teacher_logits_max
    teacher_denominator = tl.log(tl.sum(tl.exp(teacher_logits_minus_max), axis=0))

    student_logits_max = tl.max(student_logits, axis=0)
    student_logits_minus_max = student_logits - student_logits_max
    student_denominator = tl.log(tl.sum(tl.exp(student_logits_minus_max), axis=0))

    kl_loss_value = tl.sum(
        tl.exp(teacher_logits_minus_max - teacher_denominator)
        * ((teacher_logits_minus_max - teacher_denominator) - (student_logits_minus_max - student_denominator)),
        axis=0,
    )

    tl.store(kl_loss_out_ptr + idx_no_vocab, kl_loss_value)


@triton.jit
def _kl_div_bwd_kernel(
    teacher_logits_ptr,
    student_logits_ptr,
    mask_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    teacher_grad_out_ptr,
    student_grad_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for RNN-T log probs. Stores result in `grad_target_scores_ptr` and `grad_blank_scores_ptr`.
    We recalculate part of the forward here to avoid using extra memory in forward.
    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    idx_no_vocab = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    mask_value = tl.load(mask_ptr + idx_no_vocab)
    if not mask_value:
        # no calculations required
        return

    # do all calculation at least in float32
    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # recalculate log-softmax in backward instead of storing it in memory
    # calculate offset in [B, T, U+1, V] tensor for the current vector with target logits
    idx_vocab_start = idx_no_vocab * num_labels
    teacher_logits_ptr += idx_vocab_start
    student_logits_ptr += idx_vocab_start
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    teacher_logits = tl.load(teacher_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    student_logits = tl.load(student_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    # stable log softmax calculation
    teacher_logits_max = tl.max(teacher_logits, axis=0)
    teacher_logits_minus_max = teacher_logits - teacher_logits_max
    teacher_denominator = tl.log(tl.sum(tl.exp(teacher_logits_minus_max), axis=0))

    student_logits_max = tl.max(student_logits, axis=0)
    student_logits_minus_max = student_logits - student_logits_max
    student_denominator = tl.log(tl.sum(tl.exp(student_logits_minus_max), axis=0))

    # TODO
    # calculate grad


class FusedKLDivTriton(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T, supporting torch.autograd.
    """

    @staticmethod
    def forward(ctx, teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor):
        """

        Args:
            ctx: ctx object for storing the context
            teacher_logits: Joint tensor of size [B, T, U+1, D]
            student_logits: Joint tensor of size [B, T, U+1, D]
            mask: mask tensor

        Returns:
            loss of size [B, T, U+1]
        """
        assert teacher_logits.is_contiguous()  # logits are huge, so here we just check if logits are contiguous
        assert student_logits.is_contiguous()  # logits are huge, so here we just check if logits are contiguous
        assert teacher_logits.shape == student_logits.shape
        # Use float64 if input is float64, otherwise float32
        use_fp64 = teacher_logits.dtype == torch.float64

        kl_loss = teacher_logits.new_zeros(teacher_logits.shape[:-1])
        # run Triton kernel
        _kl_div_fwd_kernel[(teacher_logits.shape[0], teacher_logits.shape[1], teacher_logits.shape[2])](
            teacher_logits_ptr=teacher_logits,
            student_logits_ptr=student_logits,
            mask_ptr=mask,
            max_source_len=teacher_logits.shape[1],
            max_target_len_plus_1=teacher_logits.shape[2],
            num_labels=teacher_logits.shape[3],
            kl_loss_out_ptr=kl_loss,
            BLOCK_SIZE=triton.next_power_of_2(teacher_logits.shape[-1]),
            USE_FP64=use_fp64,
        )

        # saving for backward
        ctx.save_for_backward(teacher_logits, student_logits, mask, use_fp64)
        return kl_loss

    @staticmethod
    def backward(ctx, grad_kl_loss):
        """
        Backward calculation for RNN-T log-probs.

        Args:
            ctx: ctx object for storing the context
            grad_target_scores: upstream gradient for targets
            grad_blank_scores:  upstream gradient for blank scores

        Returns:
            gradient for logits, None for all other arguments for `forward`
        """
        (teacher_logits, student_logits, mask, use_fp64) = ctx.saved_tensors
        teacher_grad_logits = torch.zeros_like(teacher_logits)
        student_grad_logits = torch.zeros_like(student_logits)
        # TODO; kernel
        _kl_div_fwd_kernel[(teacher_logits.shape[0], teacher_logits.shape[1], teacher_logits.shape[2])](
            teacher_logits_ptr=teacher_logits,
            student_logits_ptr=student_logits,
            mask_ptr=mask,
            max_source_len=teacher_logits.shape[1],
            max_target_len_plus_1=teacher_logits.shape[2],
            num_labels=teacher_logits.shape[3],
            teacher_grad_out_ptr=teacher_grad_logits,
            student_grad_out_ptr=student_grad_logits,
            BLOCK_SIZE=triton.next_power_of_2(teacher_logits.shape[-1]),
            USE_FP64=use_fp64,
        )
        return teacher_grad_logits, student_grad_logits, None


def kl_loss_triton(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    symmetrical: bool = False,
) -> torch.Tensor:
    """
    Memory-efficient implementation of kl-div loss for RNN-T in Triton

    Args:
        teacher_logits: Joint tensor of size [B, T, U+1, D]
        student_logits: Joint tensor of size [B, T, U+1, D]
        mask: mask tensor [B, T, U+1]
        symmetrical: if loss is symmetrical

    Returns:
        tensor of size [B, T, U+1] with consistency loss
    """

    kl_loss = FusedKLDivTriton.apply(teacher_logits.detach(), student_logits, mask)
    if symmetrical:
        kl_loss = 0.5 * (kl_loss + FusedKLDivTriton.apply(student_logits.detach(), teacher_logits, mask))

    return kl_loss
