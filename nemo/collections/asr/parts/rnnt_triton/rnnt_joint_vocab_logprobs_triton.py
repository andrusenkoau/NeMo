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


def _build_flattened_batch_indices(
    device: torch.device,
    batch_size: int,
    src_max_length: int,
    tgt_max_length_plus_1: int,
):
    flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
    flattened_batch_offsets = torch.arange(flattened_batch_size, device=device, dtype=torch.int64)
    source_target_block_size = src_max_length * tgt_max_length_plus_1
    batch_indices = torch.div(flattened_batch_offsets, source_target_block_size, rounding_mode="floor")
    batch_offsets = flattened_batch_offsets - batch_indices * source_target_block_size
    source_indices = torch.div(batch_offsets, tgt_max_length_plus_1, rounding_mode="floor")
    target_indices = batch_offsets - source_indices * tgt_max_length_plus_1

    return (
        batch_indices.to(torch.int32),
        source_indices.to(torch.int32),
        target_indices.to(torch.int32),
    )


@triton.jit
def _log_add_exp(log_probs_1, log_probs_2):
    max_score = tl.maximum(log_probs_1, log_probs_2)
    return max_score + tl.log(tl.exp(log_probs_1 - max_score) + tl.exp(log_probs_2 - max_score))


@triton.jit
def _rnnt_joint_vocab_fwd_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_INT64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    int_dtype = tl.int64 if USE_INT64 else tl.int32
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    flattened_batch_block_index = tl.program_id(axis=0).to(int_dtype)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    hidden_offsets = tl.arange(0, HIDDEN_BLOCK)

    log_sum_exp_score = tl.full([FLATTENED_BATCH_BLOCK], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)
    target_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(
        targets_ptr + batch_index * max_target_len + target_index,
        mask=flattened_batch_valid_mask & target_label_mask,
        other=0,
    )

    for vocab_start_i32 in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_start = vocab_start_i32.to(int_dtype)
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size
        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

        block_logits = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(int_dtype)
            hidden_mask = (hidden_start + hidden_offsets) < hidden_dim

            hidden_ptrs = (
                joint_hidden_ptr
                + flattened_batch_offsets[:, None] * hidden_dim
                + hidden_start
                + hidden_offsets[None, :]
            )
            hidden_chunk = tl.load(
                hidden_ptrs,
                mask=output_blank_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            weight_chunk = tl.load(
                weight_ptr + vocab_offsets[:, None] * hidden_dim + hidden_start + hidden_offsets[None, :],
                mask=vocab_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            if USE_FP64:
                block_logits += tl.sum(hidden_chunk[:, None, :] * weight_chunk[None, :, :], axis=-1)
            elif USE_HIGH_PRECISION:
                block_logits += tl.dot(hidden_chunk, weight_chunk.T, input_precision="ieee").to(compute_dtype)
            else:
                block_logits += tl.dot(hidden_chunk, weight_chunk.T).to(compute_dtype)

        block_logits = tl.where(vocab_mask[None, :], block_logits, -float("inf"))
        block_logits_max = tl.max(block_logits, axis=-1)
        block_lse = tl.log(tl.sum(tl.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max
        log_sum_exp_score = _log_add_exp(log_sum_exp_score, block_lse)

        blank_logits += tl.sum(tl.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)
        target_logits += tl.sum(tl.where(vocab_offsets[None, :] == targets[:, None], block_logits, 0.0), axis=-1)

    tl.store(
        blank_logprobs_out_ptr + flattened_batch_offsets,
        blank_logits - log_sum_exp_score,
        mask=output_blank_mask,
    )

    tl.store(
        target_logprobs_out_ptr + flattened_batch_offsets,
        target_logits - log_sum_exp_score,
        mask=output_target_mask,
    )
    tl.store(
        log_sum_exp_out_ptr + flattened_batch_offsets,
        log_sum_exp_score,
        mask=output_blank_mask,
    )


@triton.jit
def _rnnt_joint_vocab_partial_hidden_bwd_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_joint_hidden_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_INT64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    flattened_batch_block_index = tl.program_id(axis=0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)

    int_dtype = tl.int64 if USE_INT64 else tl.int32
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )
    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    hidden_offsets = tl.arange(0, HIDDEN_BLOCK)

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(
        targets_ptr + batch_index * max_target_len + target_index,
        mask=output_target_mask,
        other=0,
    )

    lse = tl.load(log_sum_exp_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(compute_dtype)
    grad_target = tl.load(grad_target_scores_ptr + flattened_batch_offsets, mask=output_target_mask, other=0.0).to(
        compute_dtype
    )
    grad_blank = tl.load(grad_blank_scores_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(
        compute_dtype
    )
    sum_grad = grad_target + grad_blank

    for vocab_start_i32 in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_start = vocab_start_i32.to(int_dtype)
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
        block_logits = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(int_dtype)
            hidden_mask = (hidden_start + hidden_offsets) < hidden_dim

            hidden_ptrs = (
                joint_hidden_ptr
                + flattened_batch_offsets[:, None] * hidden_dim
                + hidden_start
                + hidden_offsets[None, :]
            )
            hidden_chunk = tl.load(
                hidden_ptrs,
                mask=output_blank_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            weight_chunk = tl.load(
                weight_ptr + vocab_offsets[:, None] * hidden_dim + hidden_start + hidden_offsets[None, :],
                mask=vocab_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            if USE_FP64:
                block_logits += tl.sum(hidden_chunk[:, None, :] * weight_chunk[None, :, :], axis=-1)
            elif USE_HIGH_PRECISION:
                block_logits += tl.dot(hidden_chunk, weight_chunk.T, input_precision="ieee").to(compute_dtype)
            else:
                block_logits += tl.dot(hidden_chunk, weight_chunk.T).to(compute_dtype)

        softmax = tl.exp(tl.where(vocab_mask[None, :], block_logits - lse[:, None], 0.0))
        grad_logits = -softmax * sum_grad[:, None]
        grad_logits += tl.where(vocab_offsets[None, :] == targets[:, None], grad_target[:, None], 0.0)
        grad_logits += tl.where((vocab_offsets == blank_id)[None, :], grad_blank[:, None], 0.0)
        grad_logits = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits, 0.0)

        grad_logits_matmul = grad_logits.to(matmul_dtype)
        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(int_dtype)
            hidden_out_offsets = hidden_start + hidden_offsets
            hidden_mask = hidden_out_offsets < hidden_dim

            weight_hidden_out = tl.load(
                weight_ptr + vocab_offsets[:, None] * hidden_dim + hidden_out_offsets[None, :],
                mask=vocab_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            if USE_FP64:
                grad_hidden_delta = tl.sum(grad_logits_matmul[:, :, None] * weight_hidden_out[None, :, :], axis=1).to(
                    compute_dtype
                )
            elif USE_HIGH_PRECISION:
                grad_hidden_delta = tl.dot(grad_logits_matmul, weight_hidden_out, input_precision="ieee").to(
                    compute_dtype
                )
            else:
                grad_hidden_delta = tl.dot(grad_logits_matmul, weight_hidden_out).to(compute_dtype)

            grad_hidden_ptrs = (
                grad_joint_hidden_out_ptr + flattened_batch_offsets[:, None] * hidden_dim + hidden_out_offsets[None, :]
            )
            grad_hidden_mask = output_blank_mask[:, None] & hidden_mask[None, :]

            old_grad_hidden = tl.load(grad_hidden_ptrs, mask=grad_hidden_mask, other=0.0).to(compute_dtype)
            tl.store(
                grad_hidden_ptrs,
                old_grad_hidden + grad_hidden_delta,
                mask=grad_hidden_mask,
            )


@triton.jit
def _rnnt_joint_vocab_partial_weight_bias_bwd_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    flattened_batch_split_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_INT64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    int_dtype = tl.int64 if USE_INT64 else tl.int32
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    vocab_block_index = tl.program_id(axis=0).to(int_dtype)
    flattened_batch_split_index = tl.program_id(axis=1).to(int_dtype)
    vocab_block_start = vocab_block_index * VOCAB_BLOCK
    vocab_offsets = vocab_block_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size

    # flattened_batch_block_index = tl.program_id(axis=0)
    # flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    # flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    # flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size
    #
    # source_target_block_size = max_src_len * max_tgt_len_plus_1
    # batch_index = flattened_batch_offsets // source_target_block_size
    # batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    # source_index = batch_offsets // max_tgt_len_plus_1
    # target_index = batch_offsets - source_index * max_tgt_len_plus_1
    #
    # source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    # target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    #

    # source_mask = source_index < source_length
    # target_valid_mask = target_index <= target_length
    # target_label_mask = target_index < target_length
    # output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    # output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    grad_bias_acc = tl.zeros((VOCAB_BLOCK,), dtype=compute_dtype)
    # grad_weight_acc = tl.zeros((VOCAB_BLOCK, hidden_dim), dtype=compute_dtype)
    is_blank_vocab_col = (vocab_offsets == blank_id) & vocab_mask

    # Atomic add into global grads
    # tl.atomic_add(
    #     grad_weight_out_ptr
    #     + vocab_offsets[:, None] * hidden_dim
    #     + hidden_offsets_full[None, :],
    #     grad_weight_acc,
    #     mask=vocab_mask[:, None] & hidden_mask_full[None, :],
    # )
    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        grad_bias_acc,
        mask=vocab_mask,
    )


class RnntJointVocabLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        joint_hidden: torch.Tensor,
        targets: torch.Tensor,
        tgt_lengths: torch.Tensor,
        src_lengths: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        blank_id: int,
        use_high_precision: bool = False,
    ):
        use_fp64 = joint_hidden.dtype == torch.float64
        float_dtype = torch.float64 if use_fp64 else torch.float32

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        FLATTENED_BATCH_BLOCK = 128
        flattened_batch_blocks = triton.cdiv(flattened_batch_size, FLATTENED_BATCH_BLOCK)
        HIDDEN_BLOCK = 64
        VOCAB_BLOCK = 64
        forward_num_stages = 1 if use_high_precision else 2
        num_warps = 4

        _rnnt_joint_vocab_fwd_kernel[(flattened_batch_blocks,)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_out_ptr=log_sum_exp_scores,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=FLATTENED_BATCH_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_INT64=True,  # use int64 indexing; currently - always, further - relax condition
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=num_warps,
            num_stages=forward_num_stages,
        )

        ctx.save_for_backward(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores)
        ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        ctx.use_high_precision = use_high_precision
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        use_high_precision = ctx.use_high_precision
        float_dtype = torch.float64 if use_fp64 else torch.float32

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        # TODO: after fixing backward kernel for joint_hidden (accumulation), use original dtype
        grad_joint_hidden = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim], dtype=float_dtype, device=device
        )

        if use_high_precision or joint_hidden.dtype != torch.bfloat16:
            hidden_bwd_vocab_block = 64
        else:
            hidden_bwd_vocab_block = 128
        hidden_bwd_flattened_batch_block = 64
        hidden_bwd_hidden_block = 64
        hidden_bwd_num_warps = 4
        hidden_bwd_num_stages = 2

        hidden_bwd_flattened_batch_blocks = triton.cdiv(flattened_batch_size, hidden_bwd_flattened_batch_block)

        _rnnt_joint_vocab_partial_hidden_bwd_kernel[(hidden_bwd_flattened_batch_blocks,)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_joint_hidden_out_ptr=grad_joint_hidden,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=hidden_bwd_flattened_batch_block,
            HIDDEN_BLOCK=hidden_bwd_hidden_block,
            VOCAB_BLOCK=hidden_bwd_vocab_block,
            USE_FP64=use_fp64,
            USE_INT64=True,  # use int64 indexing; currently - always, further - relax condition
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=hidden_bwd_num_warps,
            num_stages=hidden_bwd_num_stages,
        )

        # grad output variables
        grad_weight = torch.zeros([vocab_size, hidden_dim], dtype=float_dtype, device=device)
        grad_bias = torch.zeros([vocab_size], dtype=float_dtype, device=device)

        # device_properties = torch.cuda.get_device_properties(device)
        HIDDEN_BLOCK = 32
        VOCAB_BLOCK = 32
        FLATTENED_BATCH_BLOCK = 32
        vocab_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK)
        FLATTENED_BATCH_SPLITS = 4
        flattened_batch_split_size = triton.cdiv(flattened_batch_size, FLATTENED_BATCH_SPLITS)

        weight_bias_num_warps = 4
        weight_bias_num_stages = 2

        _rnnt_joint_vocab_partial_weight_bias_bwd_kernel[(vocab_blocks, FLATTENED_BATCH_SPLITS)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_weight_out_ptr=grad_weight,
            grad_bias_out_ptr=grad_bias,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            flattened_batch_split_size=flattened_batch_split_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=FLATTENED_BATCH_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_INT64=True,  # use int64 indexing; currently - always, further - relax condition
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=weight_bias_num_warps,
            num_stages=weight_bias_num_stages,
        )

        # convert grad to desired dtype
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)
        grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)

        return grad_joint_hidden, None, None, None, grad_weight, grad_bias, None, None


def rnnt_joint_vocab_logprobs_triton(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision: bool = False,
):
    target_logprobs, blank_logprobs = RnntJointVocabLogProbs.apply(
        joint_hidden,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        use_high_precision,
    )
    return target_logprobs, blank_logprobs
