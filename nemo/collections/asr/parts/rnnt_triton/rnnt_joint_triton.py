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


from nemo.collections.asr.parts.rnnt_triton.utils_triton import log_add_exp


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
    lse_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    joint_dim: int,
    vocab_size: int,
    blank_id: int,
    ENCODER_BLOCK: tl.constexpr,
    PREDICTOR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    """
    Forward kernel for fused RNN-T Joint + log-softmax.

    Each program handles a tile of [ENCODER_BLOCK, PREDICTOR_BLOCK] positions.
    For each tile:
    1. Loop over vocab chunks (outer), and hidden chunks (inner) to compute logits
    2. Online log-softmax across vocab chunks
    3. Extract target/blank log-probs

    Calculations are performed in float32 or float64 based on USE_FP64.
    When USE_HIGH_PRECISION is False, tl.dot uses TF32 (faster but ~10-bit mantissa).
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_block_i = tl.program_id(axis=1).to(tl.int64)
    target_block_i = tl.program_id(axis=2).to(tl.int64)
    source_i_start = source_block_i * ENCODER_BLOCK
    target_i_start = target_block_i * PREDICTOR_BLOCK

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i_start >= source_len or target_i_start > target_len:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype

    source_offsets = source_i_start + tl.arange(0, ENCODER_BLOCK)
    target_offsets = target_i_start + tl.arange(0, PREDICTOR_BLOCK)
    source_mask = source_offsets < source_len
    target_valid_mask = target_offsets <= target_len  # blank is valid at u == target_len
    target_label_mask = target_offsets < target_len  # target labels exist at u < target_len
    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK

    # Batch base pointers (source_offsets/target_offsets are absolute indices)
    enc_batch_base = batch_i * max_src_len * joint_dim
    pred_batch_base = batch_i * max_tgt_len_plus_1 * joint_dim

    vocab_chunk_offsets = tl.arange(0, VOCAB_BLOCK)
    d_offsets = tl.arange(0, HIDDEN_BLOCK)

    # Initialize log-sum-exp accumulator, blank and target logits
    log_sum_exp_score = tl.full([NUM_TILE_ELEMENTS], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)
    target_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)

    # Load target labels with batch offset
    max_tgt_len = max_tgt_len_plus_1 - 1
    targets = tl.load(targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0)
    targets_expanded = targets[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])

    # Outer loop over vocab chunks
    for v_start_i32 in tl.range(0, vocab_size, VOCAB_BLOCK):
        v_start = v_start_i32.to(tl.int64)
        v_offsets = v_start + vocab_chunk_offsets
        v_mask = v_offsets < vocab_size

        bias_chunk = tl.load(bias_ptr + v_offsets, mask=v_mask, other=0.0).to(compute_dtype)

        # Accumulate logits for this vocab chunk across hidden dimension
        logits_acc = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)

        # Inner loop over hidden dimension chunks
        for d_start_i32 in tl.range(0, joint_dim, HIDDEN_BLOCK):
            d_start = d_start_i32.to(tl.int64)
            d_mask = (d_start + d_offsets) < joint_dim

            # Load enc/pred for this hidden chunk
            enc_chunk = tl.load(
                encoder_output_ptr
                + enc_batch_base
                + source_offsets[:, None] * joint_dim
                + d_start
                + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
            )  # [ENC_CHUNK, D_CHUNK]
            pred_chunk = tl.load(
                predictor_output_ptr
                + pred_batch_base
                + target_offsets[:, None] * joint_dim
                + d_start
                + d_offsets[None, :],
                mask=target_valid_mask[:, None] & d_mask[None, :],
                other=0.0,
            )  # [PRED_CHUNK, D_CHUNK]

            # hidden = relu(enc + pred) → [ENC, PRED, D_CHUNK] → [TILE, D_CHUNK]
            hidden_chunk = (
                tl.maximum(
                    enc_chunk[:, None, :] + pred_chunk[None, :, :],
                    0.0,
                )
                .to(matmul_dtype)
                .reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])
            )

            # Load weight sub-block [V_CHUNK, D_CHUNK]
            w_chunk = tl.load(
                weight_ptr + v_offsets[:, None] * joint_dim + d_start + d_offsets[None, :],
                mask=v_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            # Accumulate matmul: logits_acc += hidden_chunk @ w_chunk^T
            if USE_FP64:
                logits_acc += tl.sum(hidden_chunk[:, None, :] * w_chunk[None, :, :], axis=-1)
            elif USE_HIGH_PRECISION:
                logits_acc = tl.dot(hidden_chunk, w_chunk.T, acc=logits_acc, input_precision="ieee").to(compute_dtype)
            else:
                logits_acc = tl.dot(hidden_chunk, w_chunk.T, acc=logits_acc).to(compute_dtype)

        # Add bias and mask invalid vocab positions
        block_logits = logits_acc + bias_chunk[None, :]  # [TILE, V_CHUNK]
        block_logits = tl.where(v_mask[None, :], block_logits, -float("inf"))

        # Online log-sum-exp
        block_logits_max = tl.max(block_logits, axis=-1)  # [TILE]
        log_sum_exp_block_score = (
            tl.log(tl.sum(tl.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max
        )
        log_sum_exp_score = log_add_exp(log_sum_exp_score, log_sum_exp_block_score)

        # Extract blank and target logits from this chunk
        blank_logits += tl.sum(tl.where((v_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)
        target_logits += tl.sum(tl.where(v_offsets[None, :] == targets_expanded[:, None], block_logits, 0.0), axis=-1)

    # Output index in [B, T, U+1] grid
    indices_grid = (batch_i * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]

    # blank logprobs (valid for all u in [0, target_len])
    tl.store(
        blank_logprobs_out_ptr + indices_grid,
        (blank_logits - log_sum_exp_score).reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=tile_valid_mask,
    )

    # Store target logprobs (valid only for u < target_len)
    target_store_mask = source_mask[:, None] & target_label_mask[None, :]
    tl.store(
        target_logprobs_out_ptr + indices_grid,
        (target_logits - log_sum_exp_score).reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=target_store_mask,
    )
    tl.store(
        lse_out_ptr + indices_grid,
        log_sum_exp_score.reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=tile_valid_mask,
    )


@triton.jit
def _rnnt_joint_partial_enc_pred_bwd_kernel(
    encoder_output_ptr,
    predictor_output_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    lse_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_encoder_partial_out_ptr,
    grad_predictor_partial_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    ENCODER_BLOCK: tl.constexpr,
    PREDICTOR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    """
    Backward kernel for fused RNN-T Joint + log-softmax.

    Computes only gradient for encoder and predictor output (inputs for Joint).
    Does not compute gradient for weight and bias.

    Each program handles a tile of [ENCODER_BLOCK, PREDICTOR_BLOCK] positions
    and writes to its own unique slice in grad_encoder_partial / grad_predictor_partial.
    No atomic operations needed.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_block_i = tl.program_id(axis=1).to(tl.int64)
    target_block_i = tl.program_id(axis=2).to(tl.int64)
    source_i_start = source_block_i * ENCODER_BLOCK
    target_i_start = target_block_i * PREDICTOR_BLOCK

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i_start >= source_len or target_i_start > target_len:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype

    source_offsets = source_i_start + tl.arange(0, ENCODER_BLOCK)
    target_offsets = target_i_start + tl.arange(0, PREDICTOR_BLOCK)
    source_mask = source_offsets < source_len
    target_valid_mask = target_offsets <= target_len
    target_label_mask = target_offsets < target_len
    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK

    enc_batch_base = batch_i * max_src_len * hidden_dim
    pred_batch_base = batch_i * max_tgt_len_plus_1 * hidden_dim

    vocab_chunk_offsets = tl.arange(0, VOCAB_BLOCK)
    d_offsets = tl.arange(0, HIDDEN_BLOCK)

    # Load target labels
    max_tgt_len = max_tgt_len_plus_1 - 1
    targets = tl.load(targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0)
    targets_expanded = targets[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])

    # Index grid for [B, T, U+1] tensors
    indices_grid = (batch_i * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]
    target_store_mask = source_mask[:, None] & target_label_mask[None, :]

    lse = tl.load(lse_ptr + indices_grid, mask=tile_valid_mask, other=0.0).reshape([NUM_TILE_ELEMENTS]).to(
        compute_dtype
    )

    # Load upstream gradients
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

    # ---- Phase 2: Compute grad_hidden, reduce to grad_enc/pred ----
    num_encoder_blocks = tl.num_programs(axis=1).to(tl.int64)
    num_predictor_blocks = tl.num_programs(axis=2).to(tl.int64)

    # Output base pointers: grad_*_partial[B, num_enc_blocks, num_pred_blocks, BLOCK, D]
    grad_enc_base = (
        batch_i * num_encoder_blocks * num_predictor_blocks * ENCODER_BLOCK * hidden_dim
        + source_block_i * num_predictor_blocks * ENCODER_BLOCK * hidden_dim
        + target_block_i * ENCODER_BLOCK * hidden_dim
    )
    grad_pred_base = (
        batch_i * num_encoder_blocks * num_predictor_blocks * PREDICTOR_BLOCK * hidden_dim
        + source_block_i * num_predictor_blocks * PREDICTOR_BLOCK * hidden_dim
        + target_block_i * PREDICTOR_BLOCK * hidden_dim
    )

    enc_row_offsets = tl.arange(0, ENCODER_BLOCK)
    pred_row_offsets = tl.arange(0, PREDICTOR_BLOCK)

    # Phase 2: V-outer double loop — compute logits once per vocab chunk
    for v_start_i32 in tl.range(0, vocab_size, VOCAB_BLOCK):
        v_start = v_start_i32.to(tl.int64)
        v_offsets = v_start + vocab_chunk_offsets
        v_mask = v_offsets < vocab_size

        # Step A: Compute logits ONCE for this vocab chunk
        bias_chunk = tl.load(bias_ptr + v_offsets, mask=v_mask, other=0.0).to(compute_dtype)
        logits_acc = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)

        for d_in_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            d_in_start = d_in_start_i32.to(tl.int64)
            d_in_mask = (d_in_start + d_offsets) < hidden_dim

            enc_in = tl.load(
                encoder_output_ptr
                + enc_batch_base
                + source_offsets[:, None] * hidden_dim
                + d_in_start
                + d_offsets[None, :],
                mask=source_mask[:, None] & d_in_mask[None, :],
                other=0.0,
            )
            pred_in = tl.load(
                predictor_output_ptr
                + pred_batch_base
                + target_offsets[:, None] * hidden_dim
                + d_in_start
                + d_offsets[None, :],
                mask=target_valid_mask[:, None] & d_in_mask[None, :],
                other=0.0,
            )
            hidden_in = (
                tl.maximum(enc_in[:, None, :] + pred_in[None, :, :], 0.0)
                .to(matmul_dtype)
                .reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])
            )

            w_in = tl.load(
                weight_ptr + v_offsets[:, None] * hidden_dim + d_in_start + d_offsets[None, :],
                mask=v_mask[:, None] & d_in_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            if USE_FP64:
                logits_acc += tl.sum(hidden_in[:, None, :] * w_in[None, :, :], axis=-1)
            elif USE_HIGH_PRECISION:
                logits_acc = tl.dot(hidden_in, w_in.T, acc=logits_acc, input_precision="ieee").to(compute_dtype)
            else:
                logits_acc = tl.dot(hidden_in, w_in.T, acc=logits_acc).to(compute_dtype)

        # Step B: Compute grad_logits
        block_logits = logits_acc + bias_chunk[None, :]
        block_logits = tl.where(v_mask[None, :], block_logits, -float("inf"))

        softmax = tl.exp(block_logits - lse[:, None])
        grad_logits = -softmax * sum_grad[:, None]

        grad_logits += tl.where(
            v_offsets[None, :] == targets_expanded[:, None],
            grad_target[:, None],
            0.0,
        )
        grad_logits += tl.where(
            (v_offsets == blank_id)[None, :],
            grad_blank[:, None],
            0.0,
        )
        grad_logits = tl.where(v_mask[None, :], grad_logits, 0.0)

        # Step C: For each d_chunk, compute grad contribution and accumulate to output
        for d_out_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            d_out_start = d_out_start_i32.to(tl.int64)
            d_out_offsets = d_out_start + d_offsets
            d_out_mask = d_out_offsets < hidden_dim

            # grad_hidden = grad_logits @ W[v, d_out]: [TILE, V_CHUNK] @ [V_CHUNK, D_CHUNK] -> [TILE, D_CHUNK]
            w_d_out = tl.load(
                weight_ptr + v_offsets[:, None] * hidden_dim + d_out_offsets[None, :],
                mask=v_mask[:, None] & d_out_mask[None, :],
                other=0.0,
            ).to(matmul_dtype)

            if USE_FP64:
                grad_hidden = tl.sum(
                    grad_logits[:, :, None].to(matmul_dtype) * w_d_out[None, :, :],
                    axis=1,
                )
            elif USE_HIGH_PRECISION:
                grad_hidden = tl.dot(
                    grad_logits.to(matmul_dtype),
                    w_d_out,
                    input_precision="ieee",
                ).to(compute_dtype)
            else:
                grad_hidden = tl.dot(
                    grad_logits.to(matmul_dtype),
                    w_d_out,
                ).to(compute_dtype)

            # Apply ReLU mask
            enc_d = tl.load(
                encoder_output_ptr + enc_batch_base + source_offsets[:, None] * hidden_dim + d_out_offsets[None, :],
                mask=source_mask[:, None] & d_out_mask[None, :],
                other=0.0,
            )
            pred_d = tl.load(
                predictor_output_ptr + pred_batch_base + target_offsets[:, None] * hidden_dim + d_out_offsets[None, :],
                mask=target_valid_mask[:, None] & d_out_mask[None, :],
                other=0.0,
            )
            relu_mask = (enc_d[:, None, :] + pred_d[None, :, :]) > 0  # [ENC, PRED, D_CHUNK]

            grad_hidden_3d = grad_hidden.reshape([ENCODER_BLOCK, PREDICTOR_BLOCK, HIDDEN_BLOCK])
            valid_3d = source_mask[:, None, None] & target_valid_mask[None, :, None] & d_out_mask[None, None, :]
            grad_pre_relu = tl.where(relu_mask & valid_3d, grad_hidden_3d, 0.0).to(compute_dtype)

            # Reduce over predictor dim -> [ENC_BLK, D_CHUNK]
            grad_enc_delta = tl.sum(grad_pre_relu, axis=1)
            # Reduce over encoder dim -> [PRED_BLK, D_CHUNK]
            grad_pred_delta = tl.sum(grad_pre_relu, axis=0)

            # Read-modify-write: accumulate partial gradients across vocab chunks
            enc_out_ptrs = (
                grad_encoder_partial_out_ptr
                + grad_enc_base
                + enc_row_offsets[:, None] * hidden_dim
                + d_out_offsets[None, :]
            )
            enc_out_mask = source_mask[:, None] & d_out_mask[None, :]
            old_enc = tl.load(enc_out_ptrs, mask=enc_out_mask, other=0.0).to(compute_dtype)
            tl.store(enc_out_ptrs, old_enc + grad_enc_delta, mask=enc_out_mask)

            pred_out_ptrs = (
                grad_predictor_partial_out_ptr
                + grad_pred_base
                + pred_row_offsets[:, None] * hidden_dim
                + d_out_offsets[None, :]
            )
            pred_out_mask = target_valid_mask[:, None] & d_out_mask[None, :]
            old_pred = tl.load(pred_out_ptrs, mask=pred_out_mask, other=0.0).to(compute_dtype)
            tl.store(pred_out_ptrs, old_pred + grad_pred_delta, mask=pred_out_mask)


@triton.jit
def _rnnt_joint_partial_weight_bias_bwd_kernel(
    encoder_output_ptr,
    predictor_output_ptr,
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
    num_encoder_blocks: int,
    num_predictor_blocks: int,
    total_tiles: int,
    ENCODER_BLOCK: tl.constexpr,
    PREDICTOR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    D_BLOCKS_PER_PROGRAM: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    V_BLOCKS_PER_PROGRAM: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    """
    Backward kernel for weight and bias gradients using split-K parallelism.

    Grid: (num_d_programs, num_v_programs, num_splits), where each v-program handles
    V_BLOCKS_PER_PROGRAM contiguous vocab blocks.
    """
    d_prog_i = tl.program_id(axis=0).to(tl.int64)
    v_prog_i = tl.program_id(axis=1).to(tl.int64)
    split_id = tl.program_id(axis=2).to(tl.int64)
    num_splits = tl.num_programs(axis=2).to(tl.int64)

    D_BLOCK: tl.constexpr = HIDDEN_BLOCK * D_BLOCKS_PER_PROGRAM
    d_start = d_prog_i * D_BLOCK
    d_abs_offsets = d_start + tl.arange(0, D_BLOCK)
    d_mask = d_abs_offsets < hidden_dim

    v_chunk_offsets = tl.arange(0, VOCAB_BLOCK)
    v_block_i0 = v_prog_i * V_BLOCKS_PER_PROGRAM
    v_start0 = v_block_i0 * VOCAB_BLOCK
    v_offsets0 = v_start0 + v_chunk_offsets
    v_mask0 = v_offsets0 < vocab_size

    if V_BLOCKS_PER_PROGRAM > 1:
        v_block_i1 = v_block_i0 + 1
        v_offsets1 = v_block_i1 * VOCAB_BLOCK + v_chunk_offsets
        v_mask1 = v_offsets1 < vocab_size

    if d_start >= hidden_dim or v_start0 >= vocab_size:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype
    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK

    # Accumulators
    grad_weight_acc0 = tl.zeros([D_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    grad_bias_acc0 = tl.zeros([VOCAB_BLOCK], dtype=compute_dtype)
    if V_BLOCKS_PER_PROGRAM > 1:
        grad_weight_acc1 = tl.zeros([D_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
        grad_bias_acc1 = tl.zeros([VOCAB_BLOCK], dtype=compute_dtype)

    # Bias gradient doesn't depend on d — only first d-program computes it
    is_first_d_program = d_prog_i == 0

    # Tile range for this split
    tiles_per_split = (total_tiles + num_splits - 1) // num_splits
    tile_start = split_id * tiles_per_split

    # Preload constants
    bias_chunk0 = tl.load(bias_ptr + v_offsets0, mask=v_mask0, other=0.0).to(compute_dtype)
    if V_BLOCKS_PER_PROGRAM > 1:
        bias_chunk1 = tl.load(bias_ptr + v_offsets1, mask=v_mask1, other=0.0).to(compute_dtype)
    d_loop_offsets = tl.arange(0, HIDDEN_BLOCK)

    for tile_offset_i32 in tl.range(0, tiles_per_split):
        tile_idx = tile_start + tile_offset_i32.to(tl.int64)
        if tile_idx < total_tiles:
            # Decompose linear tile index -> (batch, enc_block, pred_block)
            pred_block_i = tile_idx % num_predictor_blocks
            temp = tile_idx // num_predictor_blocks
            enc_block_i = temp % num_encoder_blocks
            batch_i = temp // num_encoder_blocks

            source_i_start = enc_block_i * ENCODER_BLOCK
            target_i_start = pred_block_i * PREDICTOR_BLOCK

            source_len = tl.load(src_lengths_ptr + batch_i)
            target_len = tl.load(tgt_lengths_ptr + batch_i)

            if source_i_start < source_len and target_i_start <= target_len:
                source_offsets = source_i_start + tl.arange(0, ENCODER_BLOCK)
                target_offsets = target_i_start + tl.arange(0, PREDICTOR_BLOCK)
                source_mask = source_offsets < source_len
                target_valid_mask = target_offsets <= target_len
                target_label_mask = target_offsets < target_len

                enc_batch_base = batch_i * max_src_len * hidden_dim
                pred_batch_base = batch_i * max_tgt_len_plus_1 * hidden_dim

                # Load target labels
                max_tgt_len = max_tgt_len_plus_1 - 1
                targets = tl.load(
                    targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0
                )
                targets_expanded = (
                    targets[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])
                )

                # Index grid for [B, T, U+1] tensors
                indices_grid = (batch_i * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[
                    None, :
                ]
                tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]
                target_store_mask = source_mask[:, None] & target_label_mask[None, :]
                tile_flat_mask = tile_valid_mask.reshape([NUM_TILE_ELEMENTS])

                # ---- Compute logits for grouped vocab blocks ----
                logits_acc0 = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)
                if V_BLOCKS_PER_PROGRAM > 1:
                    logits_acc1 = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype)

                for d_in_start_i32 in tl.range(0, hidden_dim, HIDDEN_BLOCK):
                    d_in_start = d_in_start_i32.to(tl.int64)
                    d_in_mask = (d_in_start + d_loop_offsets) < hidden_dim

                    enc_in = tl.load(
                        encoder_output_ptr
                        + enc_batch_base
                        + source_offsets[:, None] * hidden_dim
                        + d_in_start
                        + d_loop_offsets[None, :],
                        mask=source_mask[:, None] & d_in_mask[None, :],
                        other=0.0,
                    )
                    pred_in = tl.load(
                        predictor_output_ptr
                        + pred_batch_base
                        + target_offsets[:, None] * hidden_dim
                        + d_in_start
                        + d_loop_offsets[None, :],
                        mask=target_valid_mask[:, None] & d_in_mask[None, :],
                        other=0.0,
                    )
                    hidden_in = (
                        tl.maximum(enc_in[:, None, :] + pred_in[None, :, :], 0.0)
                        .to(matmul_dtype)
                        .reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])
                    )

                    # Logits accumulation for v_block0 (matmul)
                    w_v0 = tl.load(
                        weight_ptr + v_offsets0[:, None] * hidden_dim + d_in_start + d_loop_offsets[None, :],
                        mask=v_mask0[:, None] & d_in_mask[None, :],
                        other=0.0,
                    ).to(matmul_dtype)

                    if USE_FP64:
                        logits_acc0 += tl.sum(hidden_in[:, None, :] * w_v0[None, :, :], axis=-1)
                    elif USE_HIGH_PRECISION:
                        logits_acc0 = tl.dot(hidden_in, w_v0.T, acc=logits_acc0, input_precision="ieee").to(
                            compute_dtype
                        )
                    else:
                        logits_acc0 = tl.dot(hidden_in, w_v0.T, acc=logits_acc0).to(compute_dtype)

                    if V_BLOCKS_PER_PROGRAM > 1:
                        w_v1 = tl.load(
                            weight_ptr + v_offsets1[:, None] * hidden_dim + d_in_start + d_loop_offsets[None, :],
                            mask=v_mask1[:, None] & d_in_mask[None, :],
                            other=0.0,
                        ).to(matmul_dtype)

                        if USE_FP64:
                            logits_acc1 += tl.sum(hidden_in[:, None, :] * w_v1[None, :, :], axis=-1)
                        elif USE_HIGH_PRECISION:
                            logits_acc1 = tl.dot(hidden_in, w_v1.T, acc=logits_acc1, input_precision="ieee").to(
                                compute_dtype
                            )
                        else:
                            logits_acc1 = tl.dot(hidden_in, w_v1.T, acc=logits_acc1).to(compute_dtype)

                lse = tl.load(lse_ptr + indices_grid, mask=tile_valid_mask, other=0.0).reshape(
                    [NUM_TILE_ELEMENTS]
                ).to(compute_dtype)

                # ---- Load upstream gradients ----
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

                # ---- Compute grad_logits for v_block0 ----
                block_logits0 = logits_acc0 + bias_chunk0[None, :]
                block_logits0 = tl.where(v_mask0[None, :], block_logits0, -float("inf"))
                # numerical guard: lse is theoretically >= logits, clamp to avoid exp overflow from rounding
                logits_minus_lse0 = tl.minimum(block_logits0 - lse[:, None], 0.0)
                softmax0 = tl.exp(logits_minus_lse0)
                grad_logits0 = -softmax0 * sum_grad[:, None]

                grad_logits0 += tl.where(
                    v_offsets0[None, :] == targets_expanded[:, None],
                    grad_target[:, None],
                    0.0,
                )
                grad_logits0 += tl.where(
                    (v_offsets0 == blank_id)[None, :],
                    grad_blank[:, None],
                    0.0,
                )
                grad_logits0 = tl.where(tile_flat_mask[:, None] & v_mask0[None, :], grad_logits0, 0.0)

                if V_BLOCKS_PER_PROGRAM > 1:
                    block_logits1 = logits_acc1 + bias_chunk1[None, :]
                    block_logits1 = tl.where(v_mask1[None, :], block_logits1, -float("inf"))
                    logits_minus_lse1 = tl.minimum(block_logits1 - lse[:, None], 0.0)
                    softmax1 = tl.exp(logits_minus_lse1)
                    grad_logits1 = -softmax1 * sum_grad[:, None]

                    grad_logits1 += tl.where(
                        v_offsets1[None, :] == targets_expanded[:, None],
                        grad_target[:, None],
                        0.0,
                    )
                    grad_logits1 += tl.where(
                        (v_offsets1 == blank_id)[None, :],
                        grad_blank[:, None],
                        0.0,
                    )
                    grad_logits1 = tl.where(tile_flat_mask[:, None] & v_mask1[None, :], grad_logits1, 0.0)

                # ---- Hidden for d-slice, accumulate grad_weight ----
                enc_d = tl.load(
                    encoder_output_ptr
                    + enc_batch_base
                    + source_offsets[:, None] * hidden_dim
                    + d_abs_offsets[None, :],
                    mask=source_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                pred_d = tl.load(
                    predictor_output_ptr
                    + pred_batch_base
                    + target_offsets[:, None] * hidden_dim
                    + d_abs_offsets[None, :],
                    mask=target_valid_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                hidden_d = tl.maximum(enc_d[:, None, :] + pred_d[None, :, :], 0.0).to(matmul_dtype).reshape(
                    [NUM_TILE_ELEMENTS, D_BLOCK]
                )

                grad_logits0_matmul = grad_logits0.to(matmul_dtype)
                if USE_FP64:
                    grad_weight_acc0 += tl.sum(
                        hidden_d[:, :, None] * grad_logits0_matmul[:, None, :],
                        axis=0,
                    )
                elif USE_HIGH_PRECISION:
                    grad_weight_acc0 = tl.dot(
                        hidden_d.T, grad_logits0_matmul, acc=grad_weight_acc0, input_precision="ieee"
                    ).to(compute_dtype)
                else:
                    grad_weight_acc0 = tl.dot(hidden_d.T, grad_logits0_matmul, acc=grad_weight_acc0).to(compute_dtype)

                if V_BLOCKS_PER_PROGRAM > 1:
                    grad_logits1_matmul = grad_logits1.to(matmul_dtype)
                    if USE_FP64:
                        grad_weight_acc1 += tl.sum(
                            hidden_d[:, :, None] * grad_logits1_matmul[:, None, :],
                            axis=0,
                        )
                    elif USE_HIGH_PRECISION:
                        grad_weight_acc1 = tl.dot(
                            hidden_d.T, grad_logits1_matmul, acc=grad_weight_acc1, input_precision="ieee"
                        ).to(compute_dtype)
                    else:
                        grad_weight_acc1 = tl.dot(hidden_d.T, grad_logits1_matmul, acc=grad_weight_acc1).to(
                            compute_dtype
                        )

                # ---- Accumulate grad_bias (only first d-program) ----
                if is_first_d_program:
                    grad_bias_acc0 += tl.sum(grad_logits0, axis=0)
                    if V_BLOCKS_PER_PROGRAM > 1:
                        grad_bias_acc1 += tl.sum(grad_logits1, axis=0)

    # ---- Store partial results ----
    # grad_weight_partial layout: [num_splits, vocab_size, hidden_dim] (matches weight [V, D])
    weight_partial_offset = split_id * vocab_size * hidden_dim
    tl.store(
        grad_weight_partial_out_ptr + weight_partial_offset + v_offsets0[:, None] * hidden_dim + d_abs_offsets[None, :],
        tl.trans(grad_weight_acc0),  # [D, V] -> [V, D]
        mask=v_mask0[:, None] & d_mask[None, :],
    )
    if V_BLOCKS_PER_PROGRAM > 1:
        tl.store(
            grad_weight_partial_out_ptr
            + weight_partial_offset
            + v_offsets1[:, None] * hidden_dim
            + d_abs_offsets[None, :],
            tl.trans(grad_weight_acc1),  # [D, V] -> [V, D]
            mask=v_mask1[:, None] & d_mask[None, :],
        )

    # grad_bias_partial layout: [num_splits, vocab_size]
    if is_first_d_program:
        bias_partial_offset = split_id * vocab_size
        tl.store(
            grad_bias_partial_out_ptr + bias_partial_offset + v_offsets0,
            grad_bias_acc0,
            mask=v_mask0,
        )
        if V_BLOCKS_PER_PROGRAM > 1:
            tl.store(
                grad_bias_partial_out_ptr + bias_partial_offset + v_offsets1,
                grad_bias_acc1,
                mask=v_mask1,
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
        use_high_precision: bool = False,
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
        lse = torch.empty_like(target_logprobs)

        VOCAB_BLOCK = 64
        HIDDEN_BLOCK = 32
        ENCODER_BLOCK = 16
        PREDICTOR_BLOCK = 16

        num_encoder_blocks = triton.cdiv(src_max_length, ENCODER_BLOCK)
        num_predictor_blocks = triton.cdiv(tgt_max_length_plus_1, PREDICTOR_BLOCK)

        _rnnt_joint_fwd_kernel[(batch_size, num_encoder_blocks, num_predictor_blocks)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
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
            joint_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            ENCODER_BLOCK=ENCODER_BLOCK,
            PREDICTOR_BLOCK=PREDICTOR_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
        )

        ctx.save_for_backward(
            encoder_output_projected,
            predictor_output_projected,
            weight,
            bias,
            targets,
            src_lengths,
            tgt_lengths,
            lse,
        )
        ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        ctx.use_high_precision = use_high_precision
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
            lse,
        ) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        use_high_precision = ctx.use_high_precision
        float_dtype = torch.float64 if use_fp64 else torch.float32

        batch_size, src_max_length, hidden_dim = encoder_output_projected.shape
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = weight.shape[0]

        VOCAB_BLOCK = 64
        HIDDEN_BLOCK = 64
        ENCODER_BLOCK = 8
        PREDICTOR_BLOCK = 8

        num_encoder_blocks = triton.cdiv(src_max_length, ENCODER_BLOCK)
        num_predictor_blocks = triton.cdiv(tgt_max_length_plus_1, PREDICTOR_BLOCK)

        device = encoder_output_projected.device
        grad_encoder_partial = torch.zeros(
            [batch_size, num_encoder_blocks, num_predictor_blocks, ENCODER_BLOCK, hidden_dim],
            dtype=float_dtype,
            device=device,
        )
        grad_predictor_partial = torch.zeros(
            [batch_size, num_encoder_blocks, num_predictor_blocks, PREDICTOR_BLOCK, hidden_dim],
            dtype=float_dtype,
            device=device,
        )

        _rnnt_joint_partial_enc_pred_bwd_kernel[(batch_size, num_encoder_blocks, num_predictor_blocks)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            lse_ptr=lse,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_encoder_partial_out_ptr=grad_encoder_partial,
            grad_predictor_partial_out_ptr=grad_predictor_partial,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            ENCODER_BLOCK=ENCODER_BLOCK,
            PREDICTOR_BLOCK=PREDICTOR_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
        )

        grad_encoder = grad_encoder_partial.sum(dim=2).view([batch_size, -1, hidden_dim])[:, :src_max_length]
        grad_predictor = grad_predictor_partial.sum(dim=1).view([batch_size, -1, hidden_dim])[
            :, :tgt_max_length_plus_1
        ]
        # Weight and bias gradients via split-K kernel
        WB_ENCODER_BLOCK = 8
        WB_PREDICTOR_BLOCK = 8
        WB_HIDDEN_BLOCK = 128
        WB_D_BLOCKS_PER_PROGRAM = 2
        WB_VOCAB_BLOCK = 32
        WB_V_BLOCKS_PER_PROGRAM = 1

        wb_num_encoder_blocks = triton.cdiv(src_max_length, WB_ENCODER_BLOCK)
        wb_num_predictor_blocks = triton.cdiv(tgt_max_length_plus_1, WB_PREDICTOR_BLOCK)
        total_tiles = batch_size * wb_num_encoder_blocks * wb_num_predictor_blocks

        num_d_programs = triton.cdiv(hidden_dim, WB_HIDDEN_BLOCK * WB_D_BLOCKS_PER_PROGRAM)
        num_v_blocks = triton.cdiv(vocab_size, WB_VOCAB_BLOCK)
        num_v_programs = triton.cdiv(num_v_blocks, WB_V_BLOCKS_PER_PROGRAM)
        num_splits = min(64, total_tiles)

        grad_weight_partial = torch.zeros([num_splits, vocab_size, hidden_dim], dtype=float_dtype, device=device)
        grad_bias_partial = torch.zeros([num_splits, vocab_size], dtype=float_dtype, device=device)

        _rnnt_joint_partial_weight_bias_bwd_kernel[(num_d_programs, num_v_programs, num_splits)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
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
            num_encoder_blocks=wb_num_encoder_blocks,
            num_predictor_blocks=wb_num_predictor_blocks,
            total_tiles=total_tiles,
            ENCODER_BLOCK=WB_ENCODER_BLOCK,
            PREDICTOR_BLOCK=WB_PREDICTOR_BLOCK,
            HIDDEN_BLOCK=WB_HIDDEN_BLOCK,
            D_BLOCKS_PER_PROGRAM=WB_D_BLOCKS_PER_PROGRAM,
            VOCAB_BLOCK=WB_VOCAB_BLOCK,
            V_BLOCKS_PER_PROGRAM=WB_V_BLOCKS_PER_PROGRAM,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=4,
            num_stages=1 if use_high_precision else 2,
        )

        grad_weight = grad_weight_partial.sum(dim=0)
        grad_bias = grad_bias_partial.sum(dim=0)

        return grad_encoder, grad_predictor, None, None, None, grad_weight, grad_bias, None, None, None, None


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
    use_high_precision: bool = False,
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
        use_high_precision,
    )
    return target_logprobs, blank_logprobs
