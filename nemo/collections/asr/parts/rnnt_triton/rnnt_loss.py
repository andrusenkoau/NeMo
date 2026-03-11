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

from nemo.core.utils.optional_libs import TRITON_AVAILABLE, triton_required
from nemo.utils import logging

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_loss_triton import rnnt_loss_triton


class TritonRnntLoss(nn.Module):
    def __init__(
        self,
        blank: int,
        fastemit_lambda: float = 0.0,
        mask_non_finite: bool = False,
        warn_non_finite: bool = False,
    ):
        """
        Init method

        Args:
            blank: blank label index
            fastemit_lambda: Float scaling factor for FastEmit regularization. Default 0.0 (disabled).
            mask_non_finite: replace non-finite per-sample losses with 0.0
            warn_non_finite: log the loss tensor when non-finite values are detected
        """
        super().__init__()
        self.blank = blank
        self.fastemit_lambda = fastemit_lambda
        self.reduction = None
        self.mask_non_finite = mask_non_finite
        self.warn_non_finite = warn_non_finite
        if not TRITON_AVAILABLE:
            logging.warning("Triton is disabled, it will result error if using the loss")

    @triton_required
    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute forward method for RNN-T.

        Args:
            acts: activations (joint tensor). NB: raw logits, not after log-softmax
            labels: target labels
            act_lens: lengths of activations
            label_lens: length of labels sequences

        Returns:
            batch of RNN-T scores (loss)
        """
        # argument names are consistent with NeMo, see RNNTLoss.forward:
        # self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)
        if acts.device.type != "cuda":
            raise NotImplementedError("Triton loss supports only CUDA inputs")
        loss_batch = rnnt_loss_triton(
            blank_id=self.blank,
            logits=acts,
            targets=labels,
            src_lengths=act_lens,
            tgt_lengths=label_lens,
            fastemit_lambda=self.fastemit_lambda,
        )
        if self.warn_non_finite or self.mask_non_finite:
            loss_finite_mask = torch.isfinite(loss_batch)
            if self.warn_non_finite and not loss_finite_mask.all():
                logging.warning(f"Loss contains NaN/inf values: {loss_batch}")
            if self.mask_non_finite:
                loss_batch = torch.where(loss_finite_mask, loss_batch, torch.zeros_like(loss_batch))

        return loss_batch
