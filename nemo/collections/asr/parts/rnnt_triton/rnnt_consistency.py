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
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import get_rnnt_mask, rnnt_logprobs
from nemo.utils.enum import PrettyStrEnum


class ConsistencyRNNTReductionType(PrettyStrEnum):
    MEAN = "mean"
    MEAN_VOLUME = "mean_volume"


def _compute_kl_loss(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    symmetrical: bool,
) -> torch.Tensor:
    """Compute masked KL divergence loss."""
    kl_s_to_t = F.kl_div(
        input=student_logprobs,
        target=teacher_logprobs.detach(),
        reduction='none',
        log_target=True,
    )
    if symmetrical:
        kl_t_to_s = F.kl_div(
            input=teacher_logprobs,
            target=student_logprobs.detach(),
            reduction='none',
            log_target=True,
        )
        return 0.5 * (kl_s_to_t + kl_t_to_s) * mask
    return kl_s_to_t * mask


def consistency_rnnt_kld(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
    symmetrical: bool = True,
    use_blank: bool = False,
    reduction: str | ConsistencyRNNTReductionType = 'mean_volume',  # 'mean' or 'mean_volume'
) -> torch.Tensor:
    """
    Compute Consistency-Regularized RNN-T KL Divergence loss using targets and (optional) blank probabilities.

    Args:
        teacher_logits: Log probabilities from teacher (offline) mode [B, T, U+1, V]
        student_logits: Log probabilities from student (streaming) mode [B, T, U+1, V]
        reduction: 'mean' (normalize by frames) or 'mean_volume'

    Returns:
        Scalar KL divergence loss
    """
    reduction = ConsistencyRNNTReductionType(reduction)
    assert teacher_logits.shape == student_logits.shape
    batch_size, src_length_max, tgt_length_max_plus_1, _ = teacher_logits.shape
    device = teacher_logits.device
    teacher_target_logprobs, teacher_blank_logprobs = rnnt_logprobs(
        logits=teacher_logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    student_target_logprobs, student_blank_logprobs = rnnt_logprobs(
        logits=student_logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    mask_nb, mask_blank = get_rnnt_mask(
        batch_size=batch_size,
        src_length_max=src_length_max,
        tgt_length_max_plus_1=tgt_length_max_plus_1,
        device=device,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    kl_loss_nb = _compute_kl_loss(
        teacher_logprobs=teacher_target_logprobs,
        student_logprobs=student_target_logprobs,
        mask=mask_nb,
        symmetrical=symmetrical,
    )
    if use_blank:
        kl_loss_blank = _compute_kl_loss(
            teacher_logprobs=teacher_blank_logprobs,
            student_logprobs=student_blank_logprobs,
            mask=mask_blank,
            symmetrical=symmetrical,
        )
    else:
        kl_loss_blank = None

    match reduction:
        case ConsistencyRNNTReductionType.MEAN:
            kl_loss_value = kl_loss_nb.sum() / mask_nb.sum().clamp(min=1)
            if use_blank:
                kl_loss_value = 0.5 * (kl_loss_value + kl_loss_blank.sum() / mask_blank.sum().clamp(min=1))
        case ConsistencyRNNTReductionType.MEAN_VOLUME:
            kl_loss_value = (kl_loss_nb.sum(dim=(1, 2)) / mask_nb.sum(dim=(1, 2)).clamp(min=1)).mean()
            if use_blank:
                kl_loss_blank_value = (
                    kl_loss_blank.sum(dim=(1, 2)) / mask_blank.sum(dim=(1, 2)).clamp(min=1)
                ).mean()
                kl_loss_value = 0.5 * (kl_loss_value + kl_loss_blank_value)
        case _:
            raise NotImplementedError(f"Unsupported reduction {reduction}")

    return kl_loss_value


class ConsistencyRNNTLoss(nn.Module):
    def __init__(
        self,
        blank_id: int,
        symmetrical: bool = True,
        use_blank: bool = False,
        reduction: str | ConsistencyRNNTReductionType = 'mean_volume',
    ):
        super().__init__()
        self.reduction = ConsistencyRNNTReductionType(reduction)
        self.use_blank = use_blank
        self.blank_id = blank_id
        self.symmetrical = symmetrical

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None,
    ):
        return consistency_rnnt_kld(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            targets=targets,
            blank_id=self.blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            symmetrical=self.symmetrical,
            use_blank=self.use_blank,
            reduction=self.reduction,
        )
