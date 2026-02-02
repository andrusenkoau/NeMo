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

import pytest
import torch
from nemo.collections.asr.parts.rnnt_triton import ConsistencyRNNTLoss


@pytest.mark.unit
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_simple(symmetrical: bool, use_blank: bool, reduction: str):
    consistency_loss = ConsistencyRNNTLoss(
        blank_id=2,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss_value = consistency_loss(
        teacher_logits=torch.rand([1, 1, 2, 3]),
        student_logits=torch.rand([1, 1, 2, 3]),
        targets=torch.ones([1, 1], dtype=torch.long),
    )
    print(loss_value)
