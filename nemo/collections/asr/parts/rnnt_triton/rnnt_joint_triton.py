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
    weight_ptr,
    bias_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_scores_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for fused RNN-T Joint + log-softmax.

    For each (b, t, u) position:
    1. Compute hidden = relu(enc[b,t,:] + pred[b,u,:]) in chunks
    2. Compute logits = hidden @ W^T + bias
    3. Compute log-softmax and extract target/blank log-probs

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i >= source_len or target_i > target_len:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # Initialize logits accumulator from bias
    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offsets < vocab_size
    logits_acc = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

    # Pointers to enc[b, t, :] and pred[b, u, :]
    enc_base = batch_i * max_src_len * hidden_dim + source_i * hidden_dim
    pred_base = batch_i * max_tgt_len_plus_1 * hidden_dim + target_i * hidden_dim

    # Loop over hidden dimension in chunks of HIDDEN_BLOCK
    for d_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        d_start = d_start_i32.to(tl.int64)
        d_offsets = tl.arange(0, HIDDEN_BLOCK)
        d_mask = (d_start + d_offsets) < hidden_dim

        enc_chunk = tl.load(encoder_output_ptr + enc_base + d_start + d_offsets, mask=d_mask, other=0.0).to(
            compute_dtype
        )
        pred_chunk = tl.load(predictor_output_ptr + pred_base + d_start + d_offsets, mask=d_mask, other=0.0).to(
            compute_dtype
        )

        # Fused add + ReLU
        hidden_chunk = tl.maximum(enc_chunk + pred_chunk, 0.0)

        # Load weight block [BLOCK_SIZE, HIDDEN_BLOCK] and accumulate matmul
        # weight layout: [vocab_size, hidden_dim], row-major
        weight_block = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_dim + d_start + d_offsets[None, :],
            mask=vocab_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(compute_dtype)

        logits_acc += tl.sum(weight_block * hidden_chunk[None, :], axis=1)

    # Mask padding positions to -inf before log-softmax
    logits_acc = tl.where(vocab_mask, logits_acc, -float("inf"))

    # Stable log-softmax
    logits_max = tl.max(logits_acc, axis=0)
    logits_shifted = logits_acc - logits_max
    log_sum_exp = tl.log(tl.sum(tl.exp(logits_shifted), axis=0)) + logits_max

    # Output index in [B, T, U+1] grid
    flat_index_grid = (batch_i * max_src_len + source_i) * max_tgt_len_plus_1 + target_i

    # Blank logprob
    blank_logit = tl.sum(tl.where(vocab_offsets == blank_id, logits_acc, 0.0))
    tl.store(blank_logprobs_out_ptr + flat_index_grid, blank_logit - log_sum_exp)
    tl.store(log_sum_exp_scores_out_ptr + flat_index_grid, log_sum_exp)

    # Target logprob (only if target_i < target_len)
    if target_i < target_len:
        max_tgt_len = max_tgt_len_plus_1 - 1
        target_id = tl.load(targets_ptr + batch_i * max_tgt_len + target_i)
        target_logit = tl.sum(tl.where(vocab_offsets == target_id, logits_acc, 0.0))
        tl.store(target_logprobs_out_ptr + flat_index_grid, target_logit - log_sum_exp)


@triton.jit
def _rnnt_joint_bwd_kernel(
    encoder_output_ptr,
    predictor_output_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_scores_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_encoder_out_ptr,
    grad_predictor_out_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for fused RNN-T Joint + log-softmax.

    Recomputes forward (logits) to avoid storing the full logits tensor,
    then backpropagates through log-softmax, linear layer, and ReLU
    to produce gradients for encoder, predictor, weight, and bias.

    Uses atomic adds for encoder/predictor/weight/bias gradients since
    multiple (b,t,u) programs write to overlapping locations.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i >= source_len or target_i > target_len:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offsets < vocab_size

    # --- Recompute forward: logits_acc (including bias) ---
    logits_acc = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

    enc_base = batch_i * max_src_len * hidden_dim + source_i * hidden_dim
    pred_base = batch_i * max_tgt_len_plus_1 * hidden_dim + target_i * hidden_dim

    for d_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        d_start = d_start_i32.to(tl.int64)
        d_offsets = tl.arange(0, HIDDEN_BLOCK)
        d_mask = (d_start + d_offsets) < hidden_dim

        enc_chunk = tl.load(encoder_output_ptr + enc_base + d_start + d_offsets, mask=d_mask, other=0.0).to(
            compute_dtype
        )
        pred_chunk = tl.load(predictor_output_ptr + pred_base + d_start + d_offsets, mask=d_mask, other=0.0).to(
            compute_dtype
        )
        hidden_chunk = tl.maximum(enc_chunk + pred_chunk, 0.0)

        weight_block = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_dim + d_start + d_offsets[None, :],
            mask=vocab_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(compute_dtype)

        logits_acc += tl.sum(weight_block * hidden_chunk[None, :], axis=1)

    # --- Compute grad_logits from log-softmax backward ---
    flat_index_grid = (batch_i * max_src_len + source_i) * max_tgt_len_plus_1 + target_i
    log_sum_exp = tl.load(log_sum_exp_scores_ptr + flat_index_grid).to(compute_dtype)

    softmax = tl.exp(logits_acc - log_sum_exp)

    blank_grad = tl.load(grad_blank_scores_ptr + flat_index_grid).to(compute_dtype)
    target_i_valid = target_i < target_len
    target_grad = tl.load(grad_target_scores_ptr + flat_index_grid, mask=target_i_valid, other=0.0).to(compute_dtype)

    max_tgt_len = max_tgt_len_plus_1 - 1
    target_id = tl.load(targets_ptr + batch_i * max_tgt_len + target_i, mask=target_i_valid, other=-1)

    # Same gradient formula as _rnnt_logprobs_bwd_kernel
    grad_base = (-softmax) * (blank_grad + target_grad)
    grad_logits = tl.where(vocab_offsets == blank_id, blank_grad + grad_base, grad_base)
    grad_logits = tl.where(vocab_offsets == target_id, target_grad + grad_base, grad_logits)

    # --- Atomic add grad_bias ---
    tl.atomic_add(grad_bias_out_ptr + vocab_offsets, grad_logits, mask=vocab_mask)

    # --- Backpropagate through linear + ReLU in D-chunked loop ---
    for d_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        d_start = d_start_i32.to(tl.int64)
        d_offsets = tl.arange(0, HIDDEN_BLOCK)
        d_mask = (d_start + d_offsets) < hidden_dim

        # Reload enc/pred to recompute relu_mask
        enc_chunk = tl.load(encoder_output_ptr + enc_base + d_start + d_offsets, mask=d_mask, other=0.0).to(
            compute_dtype
        )
        pred_chunk = tl.load(predictor_output_ptr + pred_base + d_start + d_offsets, mask=d_mask, other=0.0).to(
            compute_dtype
        )
        hidden_pre_relu = enc_chunk + pred_chunk
        relu_mask = hidden_pre_relu > 0.0
        hidden_chunk = tl.where(relu_mask, hidden_pre_relu, 0.0)

        # Load weight block for this D chunk
        weight_block = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_dim + d_start + d_offsets[None, :],
            mask=vocab_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(compute_dtype)

        # grad_hidden_chunk = grad_logits^T @ weight_block -> [HIDDEN_BLOCK]
        grad_hidden_chunk = tl.sum(grad_logits[:, None] * weight_block, axis=0)

        # Apply ReLU gradient
        grad_pre_relu = tl.where(relu_mask, grad_hidden_chunk, 0.0)

        # Atomic add to encoder gradient: grad_encoder[b, t, d]
        tl.atomic_add(grad_encoder_out_ptr + enc_base + d_start + d_offsets, grad_pre_relu, mask=d_mask)

        # Atomic add to predictor gradient: grad_predictor[b, u, d]
        tl.atomic_add(grad_predictor_out_ptr + pred_base + d_start + d_offsets, grad_pre_relu, mask=d_mask)

        # Atomic add to weight gradient: grad_weight[v, d] += grad_logits[v] * hidden_chunk[d]
        grad_weight_chunk = grad_logits[:, None] * hidden_chunk[None, :]
        tl.atomic_add(
            grad_weight_out_ptr + vocab_offsets[:, None] * hidden_dim + d_start + d_offsets[None, :],
            grad_weight_chunk,
            mask=vocab_mask[:, None] & d_mask[None, :],
        )


class RnntJointLogProbs(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T, supporting torch.autograd.
    Fuses Joint network (linear + relu + linear) with log-softmax to avoid materializing large intermediate tensors.
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
        if activation != "relu":
            raise NotImplementedError("Only relu activation is supported")

        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported yet")

        use_fp64 = encoder_output_projected.dtype == torch.float64
        float_dtype = torch.float64 if use_fp64 else torch.float32

        encoder_output_projected = encoder_output_projected.contiguous()
        predictor_output_projected = predictor_output_projected.contiguous()
        targets = targets.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        device = encoder_output_projected.device
        batch_size, src_max_length, hidden_dim = encoder_output_projected.shape
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = weight.shape[0]

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.zeros_like(target_logprobs)

        BLOCK_SIZE = triton.next_power_of_2(vocab_size)
        HIDDEN_BLOCK = 32

        _rnnt_joint_fwd_kernel[(batch_size, src_max_length, tgt_max_length_plus_1)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_scores_out_ptr=log_sum_exp_scores,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            BLOCK_SIZE=BLOCK_SIZE,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            USE_FP64=use_fp64,
        )

        ctx.save_for_backward(
            encoder_output_projected,
            predictor_output_projected,
            weight,
            bias,
            targets,
            src_lengths,
            tgt_lengths,
            log_sum_exp_scores,
        )
        ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (
            encoder_output_projected,
            predictor_output_projected,
            weight,
            bias,
            targets,
            src_lengths,
            tgt_lengths,
            log_sum_exp_scores,
        ) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        # float_dtype = torch.float64 if use_fp64 else torch.float32

        batch_size, src_max_length, hidden_dim = encoder_output_projected.shape
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = weight.shape[0]

        grad_encoder = torch.zeros_like(encoder_output_projected, dtype=torch.float32)
        grad_predictor = torch.zeros_like(predictor_output_projected, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(vocab_size)
        HIDDEN_BLOCK = 32

        _rnnt_joint_bwd_kernel[(batch_size, src_max_length, tgt_max_length_plus_1)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_scores_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_encoder_out_ptr=grad_encoder,
            grad_predictor_out_ptr=grad_predictor,
            grad_weight_out_ptr=grad_weight,
            grad_bias_out_ptr=grad_bias,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            BLOCK_SIZE=BLOCK_SIZE,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            USE_FP64=use_fp64,
        )

        return grad_encoder, grad_predictor, None, None, None, grad_weight, grad_bias, None, None, None


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
