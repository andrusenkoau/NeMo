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
    lse_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    SOURCE_BLOCK: tl.constexpr,
    TARGET_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    batch_index = tl.program_id(axis=0).to(tl.int64)
    source_block_index = tl.program_id(axis=1).to(tl.int64)
    target_block_index = tl.program_id(axis=2).to(tl.int64)
    source_index_start = source_block_index * SOURCE_BLOCK
    target_index_start = target_block_index * TARGET_BLOCK

    source_length = tl.load(src_lengths_ptr + batch_index)
    target_length = tl.load(tgt_lengths_ptr + batch_index)

    if source_index_start >= source_length or target_index_start > target_length:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )
    NUM_TILE_ELEMENTS: tl.constexpr = SOURCE_BLOCK * TARGET_BLOCK

    source_offsets = source_index_start + tl.arange(0, SOURCE_BLOCK)
    target_offsets = target_index_start + tl.arange(0, TARGET_BLOCK)
    source_mask = source_offsets < source_length
    target_valid_mask = target_offsets <= target_length
    target_label_mask = target_offsets < target_length

    batch_hidden_base = batch_index * max_src_len * max_tgt_len_plus_1 * hidden_dim
    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    hidden_offsets = tl.arange(0, HIDDEN_BLOCK)

    log_sum_exp_score = tl.full([NUM_TILE_ELEMENTS], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)
    target_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(targets_ptr + batch_index * max_target_len + target_offsets, mask=target_label_mask, other=0)
    targets_expanded = targets[None, :].broadcast_to([SOURCE_BLOCK, TARGET_BLOCK]).reshape([NUM_TILE_ELEMENTS])

    for vocab_start_i32 in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_start = vocab_start_i32.to(tl.int64)
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size
        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

        logits_acc = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)
        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(tl.int64)
            hidden_mask = (hidden_start + hidden_offsets) < hidden_dim

            hidden_ptrs = (
                joint_hidden_ptr
                + batch_hidden_base
                + source_offsets[:, None, None] * max_tgt_len_plus_1 * hidden_dim
                + target_offsets[None, :, None] * hidden_dim
                + hidden_start
                + hidden_offsets[None, None, :]
            )
            hidden_chunk = tl.load(
                hidden_ptrs,
                mask=source_mask[:, None, None] & target_valid_mask[None, :, None] & hidden_mask[None, None, :],
                other=0.0,
            ).to(matmul_dtype)
            hidden_chunk = hidden_chunk.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])

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
        target_logits += tl.sum(
            tl.where(vocab_offsets[None, :] == targets_expanded[:, None], block_logits, 0.0), axis=-1
        )

    indices_grid = (batch_index * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]

    tl.store(
        blank_logprobs_out_ptr + indices_grid,
        (blank_logits - log_sum_exp_score).reshape([SOURCE_BLOCK, TARGET_BLOCK]),
        mask=tile_valid_mask,
    )

    target_store_mask = source_mask[:, None] & target_label_mask[None, :]
    tl.store(
        target_logprobs_out_ptr + indices_grid,
        (target_logits - log_sum_exp_score).reshape([SOURCE_BLOCK, TARGET_BLOCK]),
        mask=target_store_mask,
    )
    tl.store(
        lse_out_ptr + indices_grid,
        log_sum_exp_score.reshape([SOURCE_BLOCK, TARGET_BLOCK]),
        mask=tile_valid_mask,
    )


@triton.jit
def _rnnt_joint_vocab_partial_hidden_bwd_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    lse_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_joint_hidden_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    SOURCE_BLOCK: tl.constexpr,
    TARGET_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    batch_index = tl.program_id(axis=0).to(tl.int64)
    source_block_index = tl.program_id(axis=1).to(tl.int64)
    target_block_index = tl.program_id(axis=2).to(tl.int64)
    source_index_start = source_block_index * SOURCE_BLOCK
    target_index_start = target_block_index * TARGET_BLOCK

    source_length = tl.load(src_lengths_ptr + batch_index)
    target_length = tl.load(tgt_lengths_ptr + batch_index)

    if source_index_start >= source_length or target_index_start > target_length:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )
    NUM_TILE_ELEMENTS: tl.constexpr = SOURCE_BLOCK * TARGET_BLOCK

    source_offsets = source_index_start + tl.arange(0, SOURCE_BLOCK)
    target_offsets = target_index_start + tl.arange(0, TARGET_BLOCK)
    source_mask = source_offsets < source_length
    target_valid_mask = target_offsets <= target_length
    target_label_mask = target_offsets < target_length

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    hidden_offsets = tl.arange(0, HIDDEN_BLOCK)
    batch_hidden_base = batch_index * max_src_len * max_tgt_len_plus_1 * hidden_dim

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(targets_ptr + batch_index * max_target_len + target_offsets, mask=target_label_mask, other=0)
    targets_expanded = targets[None, :].broadcast_to([SOURCE_BLOCK, TARGET_BLOCK]).reshape([NUM_TILE_ELEMENTS])

    indices_grid = (batch_index * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]
    target_store_mask = source_mask[:, None] & target_label_mask[None, :]
    tile_flat_mask = tile_valid_mask.reshape([NUM_TILE_ELEMENTS])

    lse = (
        tl.load(lse_ptr + indices_grid, mask=tile_valid_mask, other=0.0).reshape([NUM_TILE_ELEMENTS]).to(compute_dtype)
    )
    grad_target = (
        tl.load(grad_target_scores_ptr + indices_grid, mask=target_store_mask, other=0.0)
        .reshape([NUM_TILE_ELEMENTS])
        .to(compute_dtype)
    )
    grad_blank = (
        tl.load(grad_blank_scores_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
        .reshape([NUM_TILE_ELEMENTS])
        .to(compute_dtype)
    )
    sum_grad = grad_target + grad_blank

    for vocab_start_i32 in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_start = vocab_start_i32.to(tl.int64)
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
        logits_acc = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)

        for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_start = hidden_start_i32.to(tl.int64)
            hidden_mask = (hidden_start + hidden_offsets) < hidden_dim

            hidden_ptrs = (
                joint_hidden_ptr
                + batch_hidden_base
                + source_offsets[:, None, None] * max_tgt_len_plus_1 * hidden_dim
                + target_offsets[None, :, None] * hidden_dim
                + hidden_start
                + hidden_offsets[None, None, :]
            )
            hidden_chunk = tl.load(
                hidden_ptrs,
                mask=source_mask[:, None, None] & target_valid_mask[None, :, None] & hidden_mask[None, None, :],
                other=0.0,
            ).to(matmul_dtype)
            hidden_chunk = hidden_chunk.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])

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
        logits_minus_lse = tl.minimum(block_logits - lse[:, None], 0.0)
        softmax = tl.exp(logits_minus_lse)
        grad_logits = -softmax * sum_grad[:, None]

        grad_logits += tl.where(vocab_offsets[None, :] == targets_expanded[:, None], grad_target[:, None], 0.0)
        grad_logits += tl.where((vocab_offsets == blank_id)[None, :], grad_blank[:, None], 0.0)
        grad_logits = tl.where(tile_flat_mask[:, None] & vocab_mask[None, :], grad_logits, 0.0)

        grad_logits_matmul = grad_logits.to(matmul_dtype)
        for hidden_out_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_out_start = hidden_out_start_i32.to(tl.int64)
            hidden_out_offsets = hidden_out_start + hidden_offsets
            hidden_out_mask = hidden_out_offsets < hidden_dim

            weight_hidden_out = tl.load(
                weight_ptr + vocab_offsets[:, None] * hidden_dim + hidden_out_offsets[None, :],
                mask=vocab_mask[:, None] & hidden_out_mask[None, :],
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
                grad_joint_hidden_out_ptr
                + batch_hidden_base
                + source_offsets[:, None, None] * max_tgt_len_plus_1 * hidden_dim
                + target_offsets[None, :, None] * hidden_dim
                + hidden_out_offsets[None, None, :]
            )
            grad_hidden_mask = (
                source_mask[:, None, None] & target_valid_mask[None, :, None] & hidden_out_mask[None, None, :]
            )

            old_grad_hidden = tl.load(grad_hidden_ptrs, mask=grad_hidden_mask, other=0.0).to(compute_dtype)
            tl.store(
                grad_hidden_ptrs,
                old_grad_hidden + grad_hidden_delta.reshape([SOURCE_BLOCK, TARGET_BLOCK, HIDDEN_BLOCK]),
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
    lse_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_weight_partial_out_ptr,
    grad_bias_partial_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    num_source_blocks: int,
    num_target_blocks: int,
    total_tiles: int,
    SOURCE_BLOCK: tl.constexpr,
    TARGET_BLOCK: tl.constexpr,
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
    NUM_TILE_ELEMENTS: tl.constexpr = SOURCE_BLOCK * TARGET_BLOCK

    grad_weight_acc_0 = tl.zeros([D_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    grad_bias_acc_0 = tl.zeros([VOCAB_BLOCK], dtype=compute_dtype)
    if V_BLOCKS_PER_PROGRAM > 1:
        grad_weight_acc_1 = tl.zeros([D_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
        grad_bias_acc_1 = tl.zeros([VOCAB_BLOCK], dtype=compute_dtype)

    is_first_d_program = d_program_index == 0

    tiles_per_split = (total_tiles + num_splits - 1) // num_splits
    tile_start = split_index * tiles_per_split

    bias_chunk_0 = tl.load(bias_ptr + vocab_offsets_0, mask=vocab_mask_0, other=0.0).to(compute_dtype)
    if V_BLOCKS_PER_PROGRAM > 1:
        bias_chunk_1 = tl.load(bias_ptr + vocab_offsets_1, mask=vocab_mask_1, other=0.0).to(compute_dtype)

    hidden_offsets = tl.arange(0, HIDDEN_BLOCK)
    for tile_offset_i32 in tl.range(0, tiles_per_split):
        tile_index = tile_start + tile_offset_i32.to(tl.int64)
        if tile_index < total_tiles:
            target_block_index = tile_index % num_target_blocks
            source_block_linear = tile_index // num_target_blocks
            source_block_index = source_block_linear % num_source_blocks
            batch_index = source_block_linear // num_source_blocks

            source_index_start = source_block_index * SOURCE_BLOCK
            target_index_start = target_block_index * TARGET_BLOCK
            source_length = tl.load(src_lengths_ptr + batch_index)
            target_length = tl.load(tgt_lengths_ptr + batch_index)

            if source_index_start < source_length and target_index_start <= target_length:
                source_offsets = source_index_start + tl.arange(0, SOURCE_BLOCK)
                target_offsets = target_index_start + tl.arange(0, TARGET_BLOCK)
                source_mask = source_offsets < source_length
                target_valid_mask = target_offsets <= target_length
                target_label_mask = target_offsets < target_length

                max_target_len = max_tgt_len_plus_1 - 1
                targets = tl.load(
                    targets_ptr + batch_index * max_target_len + target_offsets, mask=target_label_mask, other=0
                )
                targets_expanded = (
                    targets[None, :].broadcast_to([SOURCE_BLOCK, TARGET_BLOCK]).reshape([NUM_TILE_ELEMENTS])
                )

                indices_grid = (
                    batch_index * max_src_len + source_offsets[:, None]
                ) * max_tgt_len_plus_1 + target_offsets[None, :]
                tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]
                target_store_mask = source_mask[:, None] & target_label_mask[None, :]
                tile_flat_mask = tile_valid_mask.reshape([NUM_TILE_ELEMENTS])

                batch_hidden_base = batch_index * max_src_len * max_tgt_len_plus_1 * hidden_dim

                logits_acc_0 = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)
                if V_BLOCKS_PER_PROGRAM > 1:
                    logits_acc_1 = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)

                for hidden_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
                    hidden_start = hidden_start_i32.to(tl.int64)
                    hidden_mask = (hidden_start + hidden_offsets) < hidden_dim

                    hidden_ptrs = (
                        joint_hidden_ptr
                        + batch_hidden_base
                        + source_offsets[:, None, None] * max_tgt_len_plus_1 * hidden_dim
                        + target_offsets[None, :, None] * hidden_dim
                        + hidden_start
                        + hidden_offsets[None, None, :]
                    )
                    hidden_in = tl.load(
                        hidden_ptrs,
                        mask=source_mask[:, None, None]
                        & target_valid_mask[None, :, None]
                        & hidden_mask[None, None, :],
                        other=0.0,
                    ).to(matmul_dtype)
                    hidden_in = hidden_in.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])

                    weight_vocab_0 = tl.load(
                        weight_ptr + vocab_offsets_0[:, None] * hidden_dim + hidden_start + hidden_offsets[None, :],
                        mask=vocab_mask_0[:, None] & hidden_mask[None, :],
                        other=0.0,
                    ).to(matmul_dtype)

                    if USE_FP64:
                        logits_acc_0 += tl.sum(hidden_in[:, None, :] * weight_vocab_0[None, :, :], axis=-1)
                    elif USE_HIGH_PRECISION:
                        logits_acc_0 = tl.dot(
                            hidden_in, weight_vocab_0.T, acc=logits_acc_0, input_precision="ieee"
                        ).to(compute_dtype)
                    else:
                        logits_acc_0 = tl.dot(hidden_in, weight_vocab_0.T, acc=logits_acc_0).to(compute_dtype)

                    if V_BLOCKS_PER_PROGRAM > 1:
                        weight_vocab_1 = tl.load(
                            weight_ptr
                            + vocab_offsets_1[:, None] * hidden_dim
                            + hidden_start
                            + hidden_offsets[None, :],
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

                lse = (
                    tl.load(lse_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
                    .reshape([NUM_TILE_ELEMENTS])
                    .to(compute_dtype)
                )
                grad_target = (
                    tl.load(grad_target_scores_ptr + indices_grid, mask=target_store_mask, other=0.0)
                    .reshape([NUM_TILE_ELEMENTS])
                    .to(compute_dtype)
                )
                grad_blank = (
                    tl.load(grad_blank_scores_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
                    .reshape([NUM_TILE_ELEMENTS])
                    .to(compute_dtype)
                )
                sum_grad = grad_target + grad_blank

                block_logits_0 = logits_acc_0 + bias_chunk_0[None, :]
                block_logits_0 = tl.where(vocab_mask_0[None, :], block_logits_0, -float("inf"))
                logits_minus_lse_0 = tl.minimum(block_logits_0 - lse[:, None], 0.0)
                softmax_0 = tl.exp(logits_minus_lse_0)
                grad_logits_0 = -softmax_0 * sum_grad[:, None]
                grad_logits_0 += tl.where(
                    vocab_offsets_0[None, :] == targets_expanded[:, None], grad_target[:, None], 0.0
                )
                grad_logits_0 += tl.where((vocab_offsets_0 == blank_id)[None, :], grad_blank[:, None], 0.0)
                grad_logits_0 = tl.where(tile_flat_mask[:, None] & vocab_mask_0[None, :], grad_logits_0, 0.0)

                if V_BLOCKS_PER_PROGRAM > 1:
                    block_logits_1 = logits_acc_1 + bias_chunk_1[None, :]
                    block_logits_1 = tl.where(vocab_mask_1[None, :], block_logits_1, -float("inf"))
                    logits_minus_lse_1 = tl.minimum(block_logits_1 - lse[:, None], 0.0)
                    softmax_1 = tl.exp(logits_minus_lse_1)
                    grad_logits_1 = -softmax_1 * sum_grad[:, None]
                    grad_logits_1 += tl.where(
                        vocab_offsets_1[None, :] == targets_expanded[:, None], grad_target[:, None], 0.0
                    )
                    grad_logits_1 += tl.where((vocab_offsets_1 == blank_id)[None, :], grad_blank[:, None], 0.0)
                    grad_logits_1 = tl.where(tile_flat_mask[:, None] & vocab_mask_1[None, :], grad_logits_1, 0.0)

                hidden_slice_ptrs = (
                    joint_hidden_ptr
                    + batch_hidden_base
                    + source_offsets[:, None, None] * max_tgt_len_plus_1 * hidden_dim
                    + target_offsets[None, :, None] * hidden_dim
                    + hidden_slice_offsets[None, None, :]
                )
                hidden_slice = tl.load(
                    hidden_slice_ptrs,
                    mask=source_mask[:, None, None]
                    & target_valid_mask[None, :, None]
                    & hidden_slice_mask[None, None, :],
                    other=0.0,
                ).to(matmul_dtype)
                hidden_slice = hidden_slice.reshape([NUM_TILE_ELEMENTS, D_BLOCK])

                grad_logits_0_matmul = grad_logits_0.to(matmul_dtype)
                if USE_FP64:
                    grad_weight_acc_0 += tl.sum(
                        hidden_slice[:, :, None] * grad_logits_0_matmul[:, None, :], axis=0
                    ).to(compute_dtype)
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

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        lse = torch.empty_like(target_logprobs)

        SOURCE_BLOCK = 16
        TARGET_BLOCK = 16
        HIDDEN_BLOCK = 32
        VOCAB_BLOCK = 64
        forward_num_stages = 1 if use_high_precision else 2

        num_source_blocks = triton.cdiv(src_max_length, SOURCE_BLOCK)
        num_target_blocks = triton.cdiv(tgt_max_length_plus_1, TARGET_BLOCK)

        _rnnt_joint_vocab_fwd_kernel[(batch_size, num_source_blocks, num_target_blocks)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            lse_out_ptr=lse,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            SOURCE_BLOCK=SOURCE_BLOCK,
            TARGET_BLOCK=TARGET_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=4,
            num_stages=forward_num_stages,
        )

        ctx.save_for_backward(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, lse)
        ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        ctx.use_high_precision = use_high_precision
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, lse) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        use_high_precision = ctx.use_high_precision
        float_dtype = torch.float64 if use_fp64 else torch.float32

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        SOURCE_BLOCK = 8
        TARGET_BLOCK = 8
        HIDDEN_BLOCK = 64
        VOCAB_BLOCK = 64

        num_source_blocks = triton.cdiv(src_max_length, SOURCE_BLOCK)
        num_target_blocks = triton.cdiv(tgt_max_length_plus_1, TARGET_BLOCK)

        grad_joint_hidden = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim], dtype=float_dtype, device=device
        )

        _rnnt_joint_vocab_partial_hidden_bwd_kernel[(batch_size, num_source_blocks, num_target_blocks)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            lse_ptr=lse,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_joint_hidden_out_ptr=grad_joint_hidden,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            SOURCE_BLOCK=SOURCE_BLOCK,
            TARGET_BLOCK=TARGET_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=4,
            num_stages=1,
        )

        WB_SOURCE_BLOCK = 8
        WB_TARGET_BLOCK = 8
        WB_HIDDEN_BLOCK = 128
        WB_D_BLOCKS_PER_PROGRAM = 2
        WB_VOCAB_BLOCK = 64
        WB_V_BLOCKS_PER_PROGRAM = 1
        wb_num_stages = 1 if use_high_precision else 2

        wb_num_source_blocks = triton.cdiv(src_max_length, WB_SOURCE_BLOCK)
        wb_num_target_blocks = triton.cdiv(tgt_max_length_plus_1, WB_TARGET_BLOCK)
        total_tiles = batch_size * wb_num_source_blocks * wb_num_target_blocks

        num_d_programs = triton.cdiv(hidden_dim, WB_HIDDEN_BLOCK * WB_D_BLOCKS_PER_PROGRAM)
        num_vocab_blocks = triton.cdiv(vocab_size, WB_VOCAB_BLOCK)
        num_vocab_programs = triton.cdiv(num_vocab_blocks, WB_V_BLOCKS_PER_PROGRAM)
        num_splits = min(64, total_tiles)

        grad_weight_partial = torch.zeros([num_splits, vocab_size, hidden_dim], dtype=float_dtype, device=device)
        grad_bias_partial = torch.zeros([num_splits, vocab_size], dtype=float_dtype, device=device)

        _rnnt_joint_vocab_partial_weight_bias_bwd_kernel[(num_d_programs, num_vocab_programs, num_splits)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            lse_ptr=lse,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_weight_partial_out_ptr=grad_weight_partial,
            grad_bias_partial_out_ptr=grad_bias_partial,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            num_source_blocks=wb_num_source_blocks,
            num_target_blocks=wb_num_target_blocks,
            total_tiles=total_tiles,
            SOURCE_BLOCK=WB_SOURCE_BLOCK,
            TARGET_BLOCK=WB_TARGET_BLOCK,
            HIDDEN_BLOCK=WB_HIDDEN_BLOCK,
            D_BLOCKS_PER_PROGRAM=WB_D_BLOCKS_PER_PROGRAM,
            VOCAB_BLOCK=WB_VOCAB_BLOCK,
            V_BLOCKS_PER_PROGRAM=WB_V_BLOCKS_PER_PROGRAM,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=4,
            num_stages=wb_num_stages,
        )

        grad_weight = grad_weight_partial.sum(dim=0)
        grad_bias = grad_bias_partial.sum(dim=0)

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
