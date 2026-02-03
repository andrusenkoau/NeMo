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

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import rnnt_logprobs


def rnnt_best_path_align(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RNN-T best path alignment using Viterbi decoding.

    Finds the maximum-likelihood path through the RNN-T lattice and returns
    frame indices where each target token was emitted.

    Lattice structure:
    - States: (t, u) where t is frames consumed (0..T), u is tokens emitted (0..U)
    - Blank arc: (t, u) → (t+1, u) with score blank_logprobs[b, t, u] - advance frame, no emit
    - Emit arc: (t, u) → (t, u+1) with score target_logprobs[b, t, u] - emit symbol, stay on frame

    Args:
        target_logprobs: Log probabilities for target labels [B, T, U+1].
            target_logprobs[b, t, u] is the log probability of emitting target u at frame t.
        blank_logprobs: Log probabilities for blank label [B, T, U+1].
            blank_logprobs[b, t, u] is the log probability of emitting blank at state (t, u).
        src_lengths: Lengths of source sequences [B]. Each value T_b indicates valid frames 0..T_b-1.
        tgt_lengths: Lengths of target sequences [B]. Each value U_b indicates valid targets 0..U_b-1.

    Returns:
        alignments: Tensor [B, U_max] where alignments[b, u] = frame index at which target u was emitted.
            For positions u >= tgt_lengths[b], the value is 0 (padding).

    Raises:
        ValueError: If any sequence has T=0 and U>0 (impossible to align).
    """
    device = target_logprobs.device
    dtype = target_logprobs.dtype
    batch_size, T_max, U_plus_1 = target_logprobs.shape
    U_max = U_plus_1 - 1  # max target length

    # Handle edge case: U == 0 (nothing to align)
    if U_max == 0:
        return torch.zeros([batch_size, 0], dtype=torch.long, device=device)

    # Validate: T=0 with U>0 is impossible
    impossible_mask = (src_lengths == 0) & (tgt_lengths > 0)
    if impossible_mask.any():
        raise ValueError("Cannot align sequences with T=0 and U>0: impossible alignment")

    # Output alignments
    alignments = torch.zeros([batch_size, U_max], dtype=torch.long, device=device)

    NEG_INF = float("-inf")

    # Process each batch element separately to handle variable lengths correctly
    for b in range(batch_size):
        T_b = src_lengths[b].item()
        U_b = tgt_lengths[b].item()

        if U_b == 0:
            continue  # Nothing to align

        # alpha[t, u] = max log-probability to reach state (t, u)
        # State (t, u) means: consumed t frames, emitted u symbols
        # We need states from (0, 0) to (T_b, U_b)
        alpha = torch.full([T_b + 1, U_b + 1], NEG_INF, dtype=dtype, device=device)
        alpha[0, 0] = 0.0  # Start state

        # Backpointers: 0 = came from blank (vertical), 1 = came from emit (horizontal)
        backptr = torch.zeros([T_b + 1, U_b + 1], dtype=torch.int8, device=device)

        # Forward pass: fill alpha row by row (frame by frame)
        for t in range(T_b + 1):
            for u in range(U_b + 1):
                if t == 0 and u == 0:
                    continue  # Start state already initialized

                # Check if we can come from blank: (t-1, u) -> (t, u)
                if t > 0:
                    blank_score = alpha[t - 1, u] + blank_logprobs[b, t - 1, u]
                    if blank_score > alpha[t, u]:
                        alpha[t, u] = blank_score
                        backptr[t, u] = 0  # blank

                # Check if we can come from emit: (t, u-1) -> (t, u)
                if u > 0 and t < T_b:  # Can only emit at frames 0..T_b-1
                    emit_score = alpha[t, u - 1] + target_logprobs[b, t, u - 1]
                    if emit_score > alpha[t, u]:
                        alpha[t, u] = emit_score
                        backptr[t, u] = 1  # emit

        # Backtracking: trace back from (T_b, U_b) to (0, 0)
        t, u = T_b, U_b
        alignment_list = []

        while u > 0:
            bp = backptr[t, u].item()
            if bp == 1:  # Came from emit: (t, u-1) -> (t, u)
                alignment_list.append(t)  # Symbol u-1 was emitted at frame t
                u -= 1
            else:  # Came from blank: (t-1, u) -> (t, u)
                t -= 1

        # Reverse to get alignments in order
        alignment_list = alignment_list[::-1]

        # Fill alignments tensor
        for i, frame_idx in enumerate(alignment_list):
            alignments[b, i] = frame_idx

    return alignments


def align_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given logits, calculate RNN-T alignment between targets and logits using Viterbi decoding.

    Finds the maximum-likelihood path through the RNN-T lattice and returns
    frame indices where each target token was emitted.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        src_lengths: optional tensor with lengths for source utterances
        tgt_lengths: optional tensor with lengths for targets

    Returns:
        Tensor of size [B, U] with frame indices for each target token.
        alignments[b, u] = frame index at which target u was emitted.
    """
    device = logits.device
    batch_size, src_length_max, tgt_length_max_plus_1, _ = logits.shape
    tgt_length_max = tgt_length_max_plus_1 - 1

    if src_lengths is None:
        src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
    if tgt_lengths is None:
        tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max, dtype=torch.long, device=device)

    target_logprobs, blank_logprobs = rnnt_logprobs(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )

    alignments = rnnt_best_path_align(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )

    return alignments
