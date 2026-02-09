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

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs_triton import rnnt_logprobs_triton


@triton.jit
def _rnnt_fwd_kernel(
    loss_batch_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T loss.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    ...


@triton.jit
def _rnnt_bwd_kernel(
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for RNN-T loss.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)


class TritonRnntLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target_logprobs: torch.Tensor,
        blank_logprobs: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ):
        """
        Args:
            ctx: ctx object for storing the context
            target_logprobs: logprobs for target labels of size [B, T, U+1]
            blank_logprobs: logprobs for blank labels of size [B, T, U+1]
            src_lengths: source lengths of size [B]
            tgt_lengths: target lengths of size [B]

        Returns:
            loss of size [B, T, U+1]
        """
        assert target_logprobs.is_contiguous()
        assert blank_logprobs.is_contiguous()
        use_fp64 = target_logprobs.dtype == torch.float64
        batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape

        loss_batch = target_logprobs.new_zeros([batch_size])
        # TODO: implement forward
        _rnnt_fwd_kernel[(batch_size,)](
            loss_batch_ptr=loss_batch,
            BLOCK_SIZE=triton.next_power_of_2(src_max_length + tgt_max_length_plus_1),
            USE_FP64=use_fp64,
        )

        ctx.save_for_backward(target_logprobs, blank_logprobs)
        ctx.use_fp64 = use_fp64
        return loss_batch

    @staticmethod
    def backward(ctx, grad_rnnt_loss):
        """ """
        (target_logprobs, blank_logprobs) = ctx.saved_tensors
        use_fp64 = ctx.use_fp64

        target_logprobs_grad = torch.zeros_like(target_logprobs)
        blank_logprobs_grad = torch.zeros_like(blank_logprobs)
        # TODO: implement backward
        # _rnnt_bwd_kernel[...](
        #     BLOCK_SIZE=triton.next_power_of_2(...),
        #     USE_FP64=use_fp64,
        # )

        return target_logprobs_grad, blank_logprobs_grad, None, None


def rnnt_loss_triton(
    blank_id: int,
    logits: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    RNN-T loss in Triton

    Args:
        blank_id: blank index
        logits: Joint tensor of size [B, T, U+1, D], raw logits (not after log-softmax)
        targets: targets of size [B, U]
        src_lengths: source lengths of size [B]
        tgt_lengths: target lengths of size [B]
    Returns:
        tensor of size [B] with RNN-T loss
    """
    target_logprobs, blank_logprobs = rnnt_logprobs_triton(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    loss_batch = TritonRnntLossFunction.apply(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)
    return loss_batch
