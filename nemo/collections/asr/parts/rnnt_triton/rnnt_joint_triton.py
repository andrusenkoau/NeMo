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


from nemo.collections.asr.parts.rnnt_triton.utils_triton import log_add_exp, matmul


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
    log_sum_exp_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
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
    batch_i = tl.program_id(axis=0)
    source_block_i = tl.program_id(axis=1)
    target_block_i = tl.program_id(axis=2)
    source_i_start = source_block_i * ENCODER_BLOCK
    target_i_start = target_block_i * PREDICTOR_BLOCK

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i_start >= source_len or target_i_start > target_len:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    source_offsets = source_i_start + tl.arange(0, ENCODER_BLOCK)
    target_offsets = target_i_start + tl.arange(0, PREDICTOR_BLOCK)
    source_mask = source_offsets < source_len
    target_valid_mask = target_offsets <= target_len  # blank is valid at u == target_len
    target_label_mask = target_offsets < target_len  # target labels exist at u < target_len
    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK

    vocab_chunk_offsets = tl.arange(0, VOCAB_BLOCK)

    # Initialize log-sum-exp accumulator, blank and target logits
    log_sum_exp_score = tl.full([NUM_TILE_ELEMENTS], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)
    target_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)

    # Load target labels with batch offset
    max_tgt_len = max_tgt_len_plus_1 - 1
    targets_predictor = tl.load(targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0)
    targets = targets_predictor[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])

    # Create block pointers for encoder, predictor, weight, and bias
    NUM_HIDDEN_ITERS: tl.constexpr = (hidden_dim + HIDDEN_BLOCK - 1) // HIDDEN_BLOCK
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    enc_block_ptr = tl.make_block_ptr(
        base=encoder_output_ptr + batch_i * max_src_len * hidden_dim,
        shape=(source_len, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(source_i_start.to(tl.int32), 0),
        block_shape=(ENCODER_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    pred_block_ptr = tl.make_block_ptr(
        base=predictor_output_ptr + batch_i * max_tgt_len_plus_1 * hidden_dim,
        shape=(target_len + 1, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(target_i_start.to(tl.int32), 0),
        block_shape=(PREDICTOR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(0, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(VOCAB_BLOCK,),
        order=(0,),
    )

    # Outer loop over vocab chunks
    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_chunk_offsets
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)

        block_logits = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

        # Inner loop over hidden dimension chunks
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            # hidden = relu(enc + pred) -> [ENC, PRED, HIDDEN_BLOCK] -> [TILE, HIDDEN_BLOCK]
            hidden_chunk = (
                tl.maximum(
                    enc_chunk[:, None, :] + pred_chunk[None, :, :],
                    0.0,
                )
                .to(matmul_dtype)
                .reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])
            )

            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            block_logits += matmul(
                hidden_chunk, weight_chunk.T, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            enc_block_ptr = tl.advance(enc_block_ptr, (0, HIDDEN_BLOCK))
            pred_block_ptr = tl.advance(pred_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # Reset hidden dim for enc/pred/weight, advance vocab for weight/bias
        enc_block_ptr = tl.advance(enc_block_ptr, (0, -HIDDEN_RESET))
        pred_block_ptr = tl.advance(pred_block_ptr, (0, -HIDDEN_RESET))
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, -HIDDEN_RESET))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

        # Mask invalid vocab positions
        block_logits = tl.where(vocab_mask[None, :], block_logits, -float("inf"))

        # Online log-sum-exp
        block_logits_max = tl.max(block_logits, axis=-1)  # [TILE]
        block_lse = tl.log(tl.sum(tl.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max
        log_sum_exp_score = log_add_exp(log_sum_exp_score, block_lse)

        # Extract blank and target logits from this chunk
        blank_logits += tl.sum(tl.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)
        target_logits += tl.sum(tl.where(vocab_offsets[None, :] == targets[:, None], block_logits, 0.0), axis=-1)

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
    output_target_mask = source_mask[:, None] & target_label_mask[None, :]
    tl.store(
        target_logprobs_out_ptr + indices_grid,
        (target_logits - log_sum_exp_score).reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=output_target_mask,
    )
    tl.store(
        log_sum_exp_out_ptr + indices_grid,
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
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_joint_hidden_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    ENCODER_BLOCK: tl.constexpr,
    PREDICTOR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    NUM_HIDDEN_BLOCKS: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    """
    Backward kernel for fused RNN-T Joint + log-softmax.

    Computes only gradient for encoder and predictor output (inputs for Joint).
    Does not compute gradient for weight and bias.

    Each program handles a tile of [ENCODER_BLOCK, PREDICTOR_BLOCK] positions
    and writes to its own unique slice in grad_joint_hidden.
    No atomic operations needed.

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    batch_i = tl.program_id(axis=0)
    source_block_i = tl.program_id(axis=1)
    target_block_i = tl.program_id(axis=2)
    source_i_start = source_block_i * ENCODER_BLOCK
    target_i_start = target_block_i * PREDICTOR_BLOCK

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i_start >= source_len or target_i_start > target_len:
        return

    source_offsets = source_i_start + tl.arange(0, ENCODER_BLOCK)
    target_offsets = target_i_start + tl.arange(0, PREDICTOR_BLOCK)
    source_mask = source_offsets < source_len
    target_valid_mask = target_offsets <= target_len
    target_label_mask = target_offsets < target_len
    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK

    vocab_chunk_offsets = tl.arange(0, VOCAB_BLOCK)

    # Load target labels with batch offset
    max_tgt_len = max_tgt_len_plus_1 - 1
    targets_predictor = tl.load(targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0)
    targets_expanded = (
        targets_predictor[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])
    )

    # Index grid for [B, T, U+1] tensors
    indices_grid = (batch_i * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]
    target_store_mask = source_mask[:, None] & target_label_mask[None, :]
    tile_flat_mask = tile_valid_mask.reshape([NUM_TILE_ELEMENTS])

    # Load upstream gradients
    lse = (
        tl.load(log_sum_exp_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
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

    # Block pointers for enc, pred, weight, bias (same as forward)
    NUM_HIDDEN_ITERS: tl.constexpr = (hidden_dim + HIDDEN_BLOCK - 1) // HIDDEN_BLOCK
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    enc_block_ptr = tl.make_block_ptr(
        base=encoder_output_ptr + batch_i * max_src_len * hidden_dim,
        shape=(source_len, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(source_i_start.to(tl.int32), 0),
        block_shape=(ENCODER_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    pred_block_ptr = tl.make_block_ptr(
        base=predictor_output_ptr + batch_i * max_tgt_len_plus_1 * hidden_dim,
        shape=(target_len + 1, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(target_i_start.to(tl.int32), 0),
        block_shape=(PREDICTOR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(0, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(VOCAB_BLOCK,),
        order=(0,),
    )

    # Gradient accumulator setup
    if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
        grad_hidden_acc = tl.zeros([NUM_TILE_ELEMENTS, NUM_HIDDEN_BLOCKS, HIDDEN_BLOCK], dtype=compute_dtype)
        hidden_blocks_offsets = tl.arange(0, NUM_HIDDEN_BLOCKS)

    # Flat indices for grad_joint_hidden addressing (int64 for large tensors)
    flat_tile_indices = indices_grid.reshape([NUM_TILE_ELEMENTS]).to(tl.int64)

    # Outer vocab loop
    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_chunk_offsets
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)
        logits_block = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

        # Inner loop 1 (forward): recompute logits = relu(enc+pred) @ weight.T + bias
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            hidden_chunk = (
                tl.maximum(enc_chunk[:, None, :] + pred_chunk[None, :, :], 0.0)
                .to(matmul_dtype)
                .reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])
            )

            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            logits_block += matmul(
                hidden_chunk, weight_chunk.T, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            enc_block_ptr = tl.advance(enc_block_ptr, (0, HIDDEN_BLOCK))
            pred_block_ptr = tl.advance(pred_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # After loop 1: enc, pred, weight all at HIDDEN_RESET
        # Do NOT reset enc/pred — loop 2 traverses backward through all three

        # Compute grad_logits (softmax backward)
        # Clamp logits - lse to <= 0 to prevent exp overflow from recomputation rounding
        # logits_minus_lse = tl.minimum(logits_block - lse[:, None], 0.0)
        probabilities_block = tl.clamp(tl.exp(logits_block - lse[:, None]), min=0.0, max=1.0)
        grad_logits_block = (
            -(sum_grad[:, None] * probabilities_block)
            + (grad_blank[:, None] * (vocab_offsets == blank_id)[None, :])
            + (grad_target[:, None] * (vocab_offsets[None, :] == targets_expanded[:, None]))
        ).to(matmul_dtype)
        grad_logits_block = tl.where(tile_flat_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)

        # Inner loop 2 (REVERSE): grad_hidden with relu backward
        for forward_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            # Decrement before loading (reverse traversal)
            enc_block_ptr = tl.advance(enc_block_ptr, (0, -HIDDEN_BLOCK))
            pred_block_ptr = tl.advance(pred_block_ptr, (0, -HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, -HIDDEN_BLOCK))

            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            # grad_hidden_delta = grad_logits @ weight [NUM_TILE_ELEMENTS, HIDDEN_BLOCK]
            grad_hidden_delta = matmul(
                grad_logits_block, weight_chunk, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            # Reload enc/pred for relu backward mask
            enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            relu_mask = (enc_chunk[:, None, :] + pred_chunk[None, :, :] > 0.0).reshape(
                [NUM_TILE_ELEMENTS, HIDDEN_BLOCK]
            )
            grad_hidden_delta = tl.where(relu_mask, grad_hidden_delta, 0.0)

            if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
                reverse_hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - forward_hidden_idx
                grad_hidden_mask = hidden_blocks_offsets == (reverse_hidden_start // HIDDEN_BLOCK)
                grad_hidden_acc += grad_hidden_delta.expand_dims(1) * grad_hidden_mask[None, :, None]
            else:
                # Load-add-store to grad_joint_hidden at the correct hidden offset
                reverse_hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - forward_hidden_idx
                hidden_d_offsets = reverse_hidden_start + tl.arange(0, HIDDEN_BLOCK)
                d_mask = hidden_d_offsets < hidden_dim

                grad_addr = (
                    grad_joint_hidden_out_ptr + flat_tile_indices[:, None] * hidden_dim + hidden_d_offsets[None, :]
                )
                store_mask = tile_flat_mask[:, None] & d_mask[None, :]
                old_grad = tl.load(grad_addr, mask=store_mask, other=0.0).to(compute_dtype)
                tl.store(
                    grad_addr,
                    (old_grad + grad_hidden_delta).to(grad_joint_hidden_out_ptr.dtype.element_ty),
                    mask=store_mask,
                )

        # After reverse loop 2: enc, pred, weight all back at hidden=0
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, 0))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

    if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
        # Write accumulated grad_hidden to global memory
        HIDDEN_DIM_MAX: tl.constexpr = NUM_HIDDEN_BLOCKS * HIDDEN_BLOCK
        hidden_offsets_full = tl.arange(0, HIDDEN_DIM_MAX)
        hidden_mask_full = hidden_offsets_full < hidden_dim
        tl.store(
            grad_joint_hidden_out_ptr + flat_tile_indices[:, None] * hidden_dim + hidden_offsets_full[None, :],
            grad_hidden_acc.reshape([NUM_TILE_ELEMENTS, HIDDEN_DIM_MAX]).to(
                grad_joint_hidden_out_ptr.dtype.element_ty
            ),
            mask=tile_flat_mask[:, None] & hidden_mask_full[None, :],
        )


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

                lse = (
                    tl.load(lse_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
                    .reshape([NUM_TILE_ELEMENTS])
                    .to(compute_dtype)
                )

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
                hidden_d = (
                    tl.maximum(enc_d[:, None, :] + pred_d[None, :, :], 0.0)
                    .to(matmul_dtype)
                    .reshape([NUM_TILE_ELEMENTS, D_BLOCK])
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
        grad_weight_partial_out_ptr
        + weight_partial_offset
        + v_offsets0[:, None] * hidden_dim
        + d_abs_offsets[None, :],
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
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        VOCAB_BLOCK = 64
        HIDDEN_BLOCK = 64
        ENCODER_BLOCK = 16
        PREDICTOR_BLOCK = 16
        forward_num_stages = 1 if use_high_precision else 2
        num_warps = 4

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
            log_sum_exp_out_ptr=log_sum_exp_scores,
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
            num_warps=num_warps,
            num_stages=forward_num_stages,
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
            log_sum_exp_scores,
        ) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        use_high_precision = ctx.use_high_precision
        float_dtype = torch.float64 if use_fp64 else torch.float32

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()

        batch_size, src_max_length, hidden_dim = encoder_output_projected.shape
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = weight.shape[0]
        device = encoder_output_projected.device

        hidden_dtype = encoder_output_projected.dtype
        if encoder_output_projected.dtype != predictor_output_projected.dtype:
            raise NotImplementedError

        if encoder_output_projected.dtype.itemsize == 2:
            # bfloat16, float16
            VOCAB_BLOCK = 128
        else:
            VOCAB_BLOCK = 64
        HIDDEN_BLOCK = 64
        ENCODER_BLOCK = 8
        PREDICTOR_BLOCK = 8

        FULL_PRECISION_JOINT_GRAD_CALC = use_high_precision  # TODO: make extra param later
        grad_joint_hidden_dtype = float_dtype if FULL_PRECISION_JOINT_GRAD_CALC else hidden_dtype

        grad_joint_hidden = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim],
            dtype=grad_joint_hidden_dtype,
            device=device,
        )
        num_encoder_blocks = triton.cdiv(src_max_length, ENCODER_BLOCK)
        num_predictor_blocks = triton.cdiv(tgt_max_length_plus_1, PREDICTOR_BLOCK)
        num_hidden_blocks = triton.next_power_of_2(triton.cdiv(hidden_dim, HIDDEN_BLOCK))
        USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR = False
        num_warps = 4
        num_stages = 2

        _rnnt_joint_partial_enc_pred_bwd_kernel[(batch_size, num_encoder_blocks, num_predictor_blocks)](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
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
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            ENCODER_BLOCK=ENCODER_BLOCK,
            PREDICTOR_BLOCK=PREDICTOR_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            NUM_HIDDEN_BLOCKS=num_hidden_blocks,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR=USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grad_encoder = grad_joint_hidden.sum(dim=2).to(encoder_output_projected.dtype)
        grad_predictor = grad_joint_hidden.sum(dim=1).to(predictor_output_projected.dtype)

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
            lse_ptr=log_sum_exp_scores,
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
