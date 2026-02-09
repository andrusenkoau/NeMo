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
    target_logprobs_ptr,
    blank_logprobs_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    alpha_ptr,
    loss_batch_ptr,
    max_src_len,
    max_tgt_len_plus_1,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIAGS: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T loss.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    NEG_INF = -1e38

    src_len = tl.load(src_lengths_ptr + batch_i).to(tl.int64)
    tgt_len = tl.load(tgt_lengths_ptr + batch_i).to(tl.int64)

    # Base offset for this batch element in [B, T, U+1] tensors
    batch_offset = batch_i * max_src_len * max_tgt_len_plus_1

    # alpha[0, 0] = 0.0
    tl.store(alpha_ptr + batch_offset, 0.0)

    num_diags = src_len + tgt_len
    t_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)

    for d in tl.range(1, MAX_DIAGS):
        d_i64 = d.to(tl.int64)
        u_offsets = d_i64 - t_offsets
        # Mask: valid positions on this diagonal
        mask = (d_i64 < num_diags) & (t_offsets < src_len) & (u_offsets >= 0) & (u_offsets <= tgt_len)

        # Blank predecessor: alpha[t-1, u] + blank_logprobs[t-1, u] (valid when t > 0)
        blank_mask = mask & (t_offsets > 0)
        blank_pred_idx = batch_offset + (t_offsets - 1) * max_tgt_len_plus_1 + u_offsets
        blank_alpha = tl.load(alpha_ptr + blank_pred_idx, mask=blank_mask, other=NEG_INF).to(compute_dtype)
        blank_lp = tl.load(blank_logprobs_ptr + blank_pred_idx, mask=blank_mask, other=0.0).to(compute_dtype)
        blank_score = tl.where(blank_mask, blank_alpha + blank_lp, NEG_INF)

        # Emit predecessor: alpha[t, u-1] + target_logprobs[t, u-1] (valid when u > 0)
        emit_mask = mask & (u_offsets > 0)
        emit_pred_idx = batch_offset + t_offsets * max_tgt_len_plus_1 + (u_offsets - 1)
        emit_alpha = tl.load(alpha_ptr + emit_pred_idx, mask=emit_mask, other=NEG_INF).to(compute_dtype)
        emit_lp = tl.load(target_logprobs_ptr + emit_pred_idx, mask=emit_mask, other=0.0).to(compute_dtype)
        emit_score = tl.where(emit_mask, emit_alpha + emit_lp, NEG_INF)

        # logaddexp
        max_score = tl.maximum(blank_score, emit_score)
        alpha_val = max_score + tl.log(tl.exp(blank_score - max_score) + tl.exp(emit_score - max_score))

        # Store alpha values
        cur_idx = batch_offset + t_offsets * max_tgt_len_plus_1 + u_offsets
        tl.store(alpha_ptr + cur_idx, alpha_val, mask=mask)
        # Barrier needed: cross-warp store-load dependency between consecutive diagonals
        tl.debug_barrier()

    # Loss = -(alpha[src_len-1, tgt_len] + blank_logprobs[src_len-1, tgt_len])
    final_idx = batch_offset + (src_len - 1) * max_tgt_len_plus_1 + tgt_len
    final_alpha = tl.load(alpha_ptr + final_idx).to(compute_dtype)
    final_blank_lp = tl.load(blank_logprobs_ptr + final_idx).to(compute_dtype)
    loss = -(final_alpha + final_blank_lp)
    tl.store(loss_batch_ptr + batch_i, loss)


@triton.jit
def _rnnt_bwd_kernel(
    target_logprobs_ptr,
    blank_logprobs_ptr,
    alpha_ptr,
    beta_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    target_logprobs_grad_ptr,
    blank_logprobs_grad_ptr,
    max_src_len,
    max_tgt_len_plus_1,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIAGS: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for RNN-T loss.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    NEG_INF = -1e38

    src_len = tl.load(src_lengths_ptr + batch_i).to(tl.int64)
    tgt_len = tl.load(tgt_lengths_ptr + batch_i).to(tl.int64)

    batch_offset = batch_i * max_src_len * max_tgt_len_plus_1

    # Initialize beta[src_len-1, tgt_len] = blank_logprobs[src_len-1, tgt_len]
    final_idx = batch_offset + (src_len - 1) * max_tgt_len_plus_1 + tgt_len
    final_blank_lp = tl.load(blank_logprobs_ptr + final_idx).to(compute_dtype)
    tl.store(beta_ptr + final_idx, final_blank_lp)

    # log_like = alpha[src_len-1, tgt_len] + blank_logprobs[src_len-1, tgt_len]
    final_alpha = tl.load(alpha_ptr + final_idx).to(compute_dtype)
    log_like = final_alpha + final_blank_lp

    # Gradient at final state: blank_grad[src_len-1, tgt_len] = -1.0
    tl.store(blank_logprobs_grad_ptr + final_idx, -1.0)

    num_diags = src_len + tgt_len
    t_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)

    # Reverse diagonal loop: d from (num_diags - 2) down to 0
    for d_rev in tl.range(0, MAX_DIAGS):
        d = num_diags - 2 - d_rev.to(tl.int64)
        u_offsets = d - t_offsets
        mask = (d >= 0) & (t_offsets < src_len) & (u_offsets >= 0) & (u_offsets <= tgt_len)

        # Blank successor: beta[t+1, u] + blank_logprobs[t, u] (valid when t+1 < src_len)
        blank_mask = mask & (t_offsets + 1 < src_len)
        blank_succ_idx = batch_offset + (t_offsets + 1) * max_tgt_len_plus_1 + u_offsets
        blank_beta = tl.load(beta_ptr + blank_succ_idx, mask=blank_mask, other=NEG_INF).to(compute_dtype)
        cur_idx = batch_offset + t_offsets * max_tgt_len_plus_1 + u_offsets
        blank_lp = tl.load(blank_logprobs_ptr + cur_idx, mask=blank_mask, other=0.0).to(compute_dtype)
        blank_score = tl.where(blank_mask, blank_beta + blank_lp, NEG_INF)

        # Emit successor: beta[t, u+1] + target_logprobs[t, u] (valid when u+1 <= tgt_len)
        emit_mask = mask & (u_offsets + 1 <= tgt_len)
        emit_succ_idx = batch_offset + t_offsets * max_tgt_len_plus_1 + (u_offsets + 1)
        emit_beta = tl.load(beta_ptr + emit_succ_idx, mask=emit_mask, other=NEG_INF).to(compute_dtype)
        emit_lp = tl.load(target_logprobs_ptr + cur_idx, mask=emit_mask, other=0.0).to(compute_dtype)
        emit_score = tl.where(emit_mask, emit_beta + emit_lp, NEG_INF)

        # Beta: logaddexp
        max_score = tl.maximum(blank_score, emit_score)
        beta_val = max_score + tl.log(tl.exp(blank_score - max_score) + tl.exp(emit_score - max_score))
        tl.store(beta_ptr + cur_idx, beta_val, mask=mask)

        # Fused gradient computation
        # blank_grad[t, u] = -exp(alpha[t, u] + beta[t+1, u] + blank_logprobs[t, u] - log_like)
        alpha_val = tl.load(alpha_ptr + cur_idx, mask=mask, other=NEG_INF).to(compute_dtype)
        blank_grad = tl.where(
            blank_mask,
            -tl.exp(alpha_val + blank_beta + blank_lp - log_like),
            0.0,
        )
        tl.store(blank_logprobs_grad_ptr + cur_idx, blank_grad, mask=mask)

        # target_grad[t, u] = -exp(alpha[t, u] + beta[t, u+1] + target_logprobs[t, u] - log_like)
        target_grad = tl.where(
            emit_mask,
            -tl.exp(alpha_val + emit_beta + emit_lp - log_like),
            0.0,
        )
        tl.store(target_logprobs_grad_ptr + cur_idx, target_grad, mask=mask)
        # Barrier needed: cross-warp store-load dependency between consecutive diagonals
        tl.debug_barrier()


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
            loss of size [B]
        """
        assert target_logprobs.is_contiguous()
        assert blank_logprobs.is_contiguous()
        use_fp64 = target_logprobs.dtype == torch.float64
        float_dtype = torch.float64 if use_fp64 else torch.float32
        batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape

        alpha = torch.full(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            fill_value=-1e38,
            dtype=float_dtype,
            device=target_logprobs.device,
        )
        loss_batch = torch.empty([batch_size], dtype=float_dtype, device=target_logprobs.device)

        BLOCK_SIZE = triton.next_power_of_2(src_max_length + tgt_max_length_plus_1)
        MAX_DIAGS = src_max_length + tgt_max_length_plus_1 - 1
        _rnnt_fwd_kernel[(batch_size,)](
            target_logprobs_ptr=target_logprobs,
            blank_logprobs_ptr=blank_logprobs,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            alpha_ptr=alpha,
            loss_batch_ptr=loss_batch,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_DIAGS=MAX_DIAGS,
            USE_FP64=use_fp64,
        )

        ctx.save_for_backward(target_logprobs, blank_logprobs, alpha, src_lengths, tgt_lengths)
        ctx.use_fp64 = use_fp64
        return loss_batch

    @staticmethod
    def backward(ctx, grad_rnnt_loss):
        """ """
        (target_logprobs, blank_logprobs, alpha, src_lengths, tgt_lengths) = ctx.saved_tensors
        use_fp64 = ctx.use_fp64
        float_dtype = torch.float64 if use_fp64 else torch.float32
        batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape

        beta = torch.full(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            fill_value=-1e38,
            dtype=float_dtype,
            device=target_logprobs.device,
        )
        target_logprobs_grad = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            dtype=float_dtype,
            device=target_logprobs.device,
        )
        blank_logprobs_grad = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            dtype=float_dtype,
            device=target_logprobs.device,
        )

        BLOCK_SIZE = triton.next_power_of_2(src_max_length + tgt_max_length_plus_1)
        MAX_DIAGS = src_max_length + tgt_max_length_plus_1 - 1
        _rnnt_bwd_kernel[(batch_size,)](
            target_logprobs_ptr=target_logprobs,
            blank_logprobs_ptr=blank_logprobs,
            alpha_ptr=alpha,
            beta_ptr=beta,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            target_logprobs_grad_ptr=target_logprobs_grad,
            blank_logprobs_grad_ptr=blank_logprobs_grad,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_DIAGS=MAX_DIAGS,
            USE_FP64=use_fp64,
        )

        # Multiply by upstream gradient
        grad_rnnt_loss = grad_rnnt_loss.to(float_dtype).view(-1, 1, 1)
        target_logprobs_grad = target_logprobs_grad * grad_rnnt_loss
        blank_logprobs_grad = blank_logprobs_grad * grad_rnnt_loss

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
