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
    USE_HIGH_PRECISION: tl.constexpr,
):
    flattened_batch_block_index = tl.program_id(axis=0).to(tl.int64)
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
        vocab_start = vocab_start_i32.to(tl.int64)
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size
        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

        logits_acc = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(tl.int64)
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
                logits_acc += tl.sum(hidden_chunk[:, None, :] * weight_chunk[None, :, :], axis=-1)
            elif USE_HIGH_PRECISION:
                logits_acc = tl.dot(hidden_chunk, weight_chunk.T, acc=logits_acc, input_precision="ieee").to(
                    compute_dtype
                )
            else:
                logits_acc = tl.dot(hidden_chunk, weight_chunk.T, acc=logits_acc).to(compute_dtype)

        block_logits = logits_acc + bias_chunk[None, :]
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
        vocab_start = vocab_start_i32.to(tl.int64)
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
        block_logits = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(tl.int64)
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
                block_logits = tl.dot(hidden_chunk, weight_chunk.T, acc=block_logits, input_precision="ieee").to(
                    compute_dtype
                )
            else:
                block_logits = tl.dot(hidden_chunk, weight_chunk.T, acc=block_logits).to(compute_dtype)

        softmax = tl.exp(tl.where(vocab_mask[None, :], block_logits - lse[:, None], 0.0))
        grad_logits = -softmax * sum_grad[:, None]
        grad_logits += tl.where(vocab_offsets[None, :] == targets[:, None], grad_target[:, None], 0.0)
        grad_logits += tl.where((vocab_offsets == blank_id)[None, :], grad_blank[:, None], 0.0)
        grad_logits = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits, 0.0)

        grad_logits_matmul = grad_logits.to(matmul_dtype)
        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(tl.int64)
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
    flattened_batch_to_batch_ptr,
    flattened_batch_to_source_ptr,
    flattened_batch_to_target_ptr,
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_weight_partial_out_ptr,
    grad_bias_partial_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    D_BLOCKS_PER_PROGRAM: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    V_BLOCKS_PER_PROGRAM: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    d_program_index = tl.program_id(axis=0).to(tl.int64)
    vocab_program_index = tl.program_id(axis=1).to(tl.int64)
    split_index = tl.program_id(axis=2).to(tl.int64)
    num_splits = tl.num_programs(axis=2).to(tl.int64)

    D_BLOCK: tl.constexpr = HIDDEN_BLOCK * D_BLOCKS_PER_PROGRAM
    hidden_slice_start = d_program_index * D_BLOCK
    hidden_slice_offsets = hidden_slice_start + tl.arange(0, D_BLOCK)
    hidden_slice_mask = hidden_slice_offsets < hidden_dim

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    vocab_block_index_0 = vocab_program_index * V_BLOCKS_PER_PROGRAM
    vocab_start_0 = vocab_block_index_0 * VOCAB_BLOCK
    vocab_offsets_0 = vocab_start_0 + vocab_offsets_in_block
    vocab_mask_0 = vocab_offsets_0 < vocab_size

    if V_BLOCKS_PER_PROGRAM > 1:
        vocab_block_index_1 = vocab_block_index_0 + 1
        vocab_offsets_1 = vocab_block_index_1 * VOCAB_BLOCK + vocab_offsets_in_block
        vocab_mask_1 = vocab_offsets_1 < vocab_size

    if hidden_slice_start >= hidden_dim or vocab_start_0 >= vocab_size:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    grad_weight_acc_0 = tl.zeros([D_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    grad_bias_acc_0 = tl.zeros([VOCAB_BLOCK], dtype=compute_dtype)
    if V_BLOCKS_PER_PROGRAM > 1:
        grad_weight_acc_1 = tl.zeros([D_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
        grad_bias_acc_1 = tl.zeros([VOCAB_BLOCK], dtype=compute_dtype)

    is_first_d_program = d_program_index == 0

    total_flattened_blocks = (flattened_batch_size + FLATTENED_BATCH_BLOCK - 1) // FLATTENED_BATCH_BLOCK
    blocks_per_split = (total_flattened_blocks + num_splits - 1) // num_splits
    block_start = split_index * blocks_per_split

    bias_chunk_0 = tl.load(bias_ptr + vocab_offsets_0, mask=vocab_mask_0, other=0.0).to(compute_dtype)
    if V_BLOCKS_PER_PROGRAM > 1:
        bias_chunk_1 = tl.load(bias_ptr + vocab_offsets_1, mask=vocab_mask_1, other=0.0).to(compute_dtype)

    hidden_offsets = tl.arange(0, HIDDEN_BLOCK)
    for block_offset_i32 in tl.range(0, blocks_per_split):
        flattened_batch_block_index = block_start + block_offset_i32.to(tl.int64)
        if flattened_batch_block_index < total_flattened_blocks:
            flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
            flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
            flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

            batch_index = tl.load(
                flattened_batch_to_batch_ptr + flattened_batch_offsets, mask=flattened_batch_valid_mask, other=0
            ).to(tl.int64)
            source_index = tl.load(
                flattened_batch_to_source_ptr + flattened_batch_offsets, mask=flattened_batch_valid_mask, other=0
            ).to(tl.int64)
            target_index = tl.load(
                flattened_batch_to_target_ptr + flattened_batch_offsets, mask=flattened_batch_valid_mask, other=0
            ).to(tl.int64)

            source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
            target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)

            source_mask = source_index < source_length
            target_valid_mask = target_index <= target_length
            target_label_mask = target_index < target_length
            output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
            output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

            max_target_len = max_tgt_len_plus_1 - 1
            targets = tl.load(
                targets_ptr + batch_index * max_target_len + target_index,
                mask=output_target_mask,
                other=0,
            )

            logits_acc_0 = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
            if V_BLOCKS_PER_PROGRAM > 1:
                logits_acc_1 = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)

            for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
                hidden_start = hidden_start_i32.to(tl.int64)
                hidden_mask = (hidden_start + hidden_offsets) < hidden_dim

                hidden_ptrs = (
                    joint_hidden_ptr
                    + flattened_batch_offsets[:, None] * hidden_dim
                    + hidden_start
                    + hidden_offsets[None, :]
                )
                hidden_in = tl.load(
                    hidden_ptrs,
                    mask=output_blank_mask[:, None] & hidden_mask[None, :],
                    other=0.0,
                ).to(matmul_dtype)

                weight_vocab_0 = tl.load(
                    weight_ptr + vocab_offsets_0[:, None] * hidden_dim + hidden_start + hidden_offsets[None, :],
                    mask=vocab_mask_0[:, None] & hidden_mask[None, :],
                    other=0.0,
                ).to(matmul_dtype)

                if USE_FP64:
                    logits_acc_0 += tl.sum(hidden_in[:, None, :] * weight_vocab_0[None, :, :], axis=-1)
                elif USE_HIGH_PRECISION:
                    logits_acc_0 = tl.dot(hidden_in, weight_vocab_0.T, acc=logits_acc_0, input_precision="ieee").to(
                        compute_dtype
                    )
                else:
                    logits_acc_0 = tl.dot(hidden_in, weight_vocab_0.T, acc=logits_acc_0).to(compute_dtype)

                if V_BLOCKS_PER_PROGRAM > 1:
                    weight_vocab_1 = tl.load(
                        weight_ptr + vocab_offsets_1[:, None] * hidden_dim + hidden_start + hidden_offsets[None, :],
                        mask=vocab_mask_1[:, None] & hidden_mask[None, :],
                        other=0.0,
                    ).to(matmul_dtype)

                    if USE_FP64:
                        logits_acc_1 += tl.sum(hidden_in[:, None, :] * weight_vocab_1[None, :, :], axis=-1)
                    elif USE_HIGH_PRECISION:
                        logits_acc_1 = tl.dot(
                            hidden_in, weight_vocab_1.T, acc=logits_acc_1, input_precision="ieee"
                        ).to(compute_dtype)
                    else:
                        logits_acc_1 = tl.dot(hidden_in, weight_vocab_1.T, acc=logits_acc_1).to(compute_dtype)

            lse = tl.load(log_sum_exp_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(
                compute_dtype
            )
            grad_target = tl.load(
                grad_target_scores_ptr + flattened_batch_offsets,
                mask=output_target_mask,
                other=0.0,
            ).to(compute_dtype)
            grad_blank = tl.load(
                grad_blank_scores_ptr + flattened_batch_offsets,
                mask=output_blank_mask,
                other=0.0,
            ).to(compute_dtype)
            sum_grad = grad_target + grad_blank

            block_logits_0 = logits_acc_0 + bias_chunk_0[None, :]
            block_logits_0 = tl.where(vocab_mask_0[None, :], block_logits_0, -float("inf"))
            logits_minus_lse_0 = tl.minimum(block_logits_0 - lse[:, None], 0.0)
            softmax_0 = tl.exp(logits_minus_lse_0)
            grad_logits_0 = -softmax_0 * sum_grad[:, None]
            grad_logits_0 += tl.where(vocab_offsets_0[None, :] == targets[:, None], grad_target[:, None], 0.0)
            grad_logits_0 += tl.where((vocab_offsets_0 == blank_id)[None, :], grad_blank[:, None], 0.0)
            grad_logits_0 = tl.where(output_blank_mask[:, None] & vocab_mask_0[None, :], grad_logits_0, 0.0)

            if V_BLOCKS_PER_PROGRAM > 1:
                block_logits_1 = logits_acc_1 + bias_chunk_1[None, :]
                block_logits_1 = tl.where(vocab_mask_1[None, :], block_logits_1, -float("inf"))
                logits_minus_lse_1 = tl.minimum(block_logits_1 - lse[:, None], 0.0)
                softmax_1 = tl.exp(logits_minus_lse_1)
                grad_logits_1 = -softmax_1 * sum_grad[:, None]
                grad_logits_1 += tl.where(vocab_offsets_1[None, :] == targets[:, None], grad_target[:, None], 0.0)
                grad_logits_1 += tl.where((vocab_offsets_1 == blank_id)[None, :], grad_blank[:, None], 0.0)
                grad_logits_1 = tl.where(output_blank_mask[:, None] & vocab_mask_1[None, :], grad_logits_1, 0.0)

            hidden_slice_ptrs = (
                joint_hidden_ptr + flattened_batch_offsets[:, None] * hidden_dim + hidden_slice_offsets[None, :]
            )
            hidden_slice = tl.load(
                hidden_slice_ptrs,
                mask=output_blank_mask[:, None] & hidden_slice_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            grad_logits_0_matmul = grad_logits_0.to(matmul_dtype)
            if USE_FP64:
                grad_weight_acc_0 += tl.sum(hidden_slice[:, :, None] * grad_logits_0_matmul[:, None, :], axis=0).to(
                    compute_dtype
                )
            elif USE_HIGH_PRECISION:
                grad_weight_acc_0 = tl.dot(
                    hidden_slice.T, grad_logits_0_matmul, acc=grad_weight_acc_0, input_precision="ieee"
                ).to(compute_dtype)
            else:
                grad_weight_acc_0 = tl.dot(hidden_slice.T, grad_logits_0_matmul, acc=grad_weight_acc_0).to(
                    compute_dtype
                )

            if V_BLOCKS_PER_PROGRAM > 1:
                grad_logits_1_matmul = grad_logits_1.to(matmul_dtype)
                if USE_FP64:
                    grad_weight_acc_1 += tl.sum(
                        hidden_slice[:, :, None] * grad_logits_1_matmul[:, None, :], axis=0
                    ).to(compute_dtype)
                elif USE_HIGH_PRECISION:
                    grad_weight_acc_1 = tl.dot(
                        hidden_slice.T, grad_logits_1_matmul, acc=grad_weight_acc_1, input_precision="ieee"
                    ).to(compute_dtype)
                else:
                    grad_weight_acc_1 = tl.dot(hidden_slice.T, grad_logits_1_matmul, acc=grad_weight_acc_1).to(
                        compute_dtype
                    )

            if is_first_d_program:
                grad_bias_acc_0 += tl.sum(grad_logits_0, axis=0)
                if V_BLOCKS_PER_PROGRAM > 1:
                    grad_bias_acc_1 += tl.sum(grad_logits_1, axis=0)

    weight_partial_offset = split_index * vocab_size * hidden_dim
    tl.store(
        grad_weight_partial_out_ptr
        + weight_partial_offset
        + vocab_offsets_0[:, None] * hidden_dim
        + hidden_slice_offsets[None, :],
        tl.trans(grad_weight_acc_0),
        mask=vocab_mask_0[:, None] & hidden_slice_mask[None, :],
    )
    if V_BLOCKS_PER_PROGRAM > 1:
        tl.store(
            grad_weight_partial_out_ptr
            + weight_partial_offset
            + vocab_offsets_1[:, None] * hidden_dim
            + hidden_slice_offsets[None, :],
            tl.trans(grad_weight_acc_1),
            mask=vocab_mask_1[:, None] & hidden_slice_mask[None, :],
        )

    if is_first_d_program:
        bias_partial_offset = split_index * vocab_size
        tl.store(
            grad_bias_partial_out_ptr + bias_partial_offset + vocab_offsets_0,
            grad_bias_acc_0,
            mask=vocab_mask_0,
        )
        if V_BLOCKS_PER_PROGRAM > 1:
            tl.store(
                grad_bias_partial_out_ptr + bias_partial_offset + vocab_offsets_1,
                grad_bias_acc_1,
                mask=vocab_mask_1,
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

        grad_joint_hidden = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim], dtype=joint_hidden.dtype, device=device
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
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=hidden_bwd_num_warps,
            num_stages=hidden_bwd_num_stages,
        )

        flattened_batch_to_batch, flattened_batch_to_source, flattened_batch_to_target = (
            _build_flattened_batch_indices(
                device=device,
                batch_size=batch_size,
                src_max_length=src_max_length,
                tgt_max_length_plus_1=tgt_max_length_plus_1,
            )
        )

        device_properties = torch.cuda.get_device_properties(device)
        partial_weight_element_size = 8 if float_dtype == torch.float64 else 4
        bytes_per_split = (
            vocab_size * hidden_dim * partial_weight_element_size + vocab_size * partial_weight_element_size
        )
        max_splits_by_memory = max(1, int((device_properties.total_memory // 8) // bytes_per_split))

        if use_high_precision:
            weight_bias_flattened_batch_block = 64
            weight_bias_hidden_block = 64
            weight_bias_d_blocks_per_program = 4
            weight_bias_vocab_block = 64
            weight_bias_v_blocks_per_program = 1
            weight_bias_num_warps = 4
            weight_bias_num_stages = 2
        else:
            weight_bias_flattened_batch_block = 64
            weight_bias_hidden_block = 128 if joint_hidden.dtype == torch.bfloat16 else 64
            weight_bias_d_blocks_per_program = 4
            weight_bias_vocab_block = 64
            weight_bias_v_blocks_per_program = 1
            weight_bias_num_warps = 8 if joint_hidden.dtype == torch.bfloat16 else 4
            weight_bias_num_stages = 2

        weight_bias_flattened_batch_blocks = triton.cdiv(flattened_batch_size, weight_bias_flattened_batch_block)
        num_splits = max(1, min(64, weight_bias_flattened_batch_blocks, max_splits_by_memory))

        grad_weight_partial = torch.zeros([num_splits, vocab_size, hidden_dim], dtype=float_dtype, device=device)
        grad_bias_partial = torch.zeros([num_splits, vocab_size], dtype=float_dtype, device=device)

        weight_bias_programs_hidden = triton.cdiv(
            hidden_dim, weight_bias_hidden_block * weight_bias_d_blocks_per_program
        )
        weight_bias_programs_vocab = triton.cdiv(
            triton.cdiv(vocab_size, weight_bias_vocab_block), weight_bias_v_blocks_per_program
        )
        _rnnt_joint_vocab_partial_weight_bias_bwd_kernel[
            (weight_bias_programs_hidden, weight_bias_programs_vocab, num_splits)
        ](
            flattened_batch_to_batch_ptr=flattened_batch_to_batch,
            flattened_batch_to_source_ptr=flattened_batch_to_source,
            flattened_batch_to_target_ptr=flattened_batch_to_target,
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_weight_partial_out_ptr=grad_weight_partial,
            grad_bias_partial_out_ptr=grad_bias_partial,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=weight_bias_flattened_batch_block,
            HIDDEN_BLOCK=weight_bias_hidden_block,
            D_BLOCKS_PER_PROGRAM=weight_bias_d_blocks_per_program,
            VOCAB_BLOCK=weight_bias_vocab_block,
            V_BLOCKS_PER_PROGRAM=weight_bias_v_blocks_per_program,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=weight_bias_num_warps,
            num_stages=weight_bias_num_stages,
        )

        grad_weight = grad_weight_partial.sum(dim=0)
        grad_bias = grad_bias_partial.sum(dim=0)
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)

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
