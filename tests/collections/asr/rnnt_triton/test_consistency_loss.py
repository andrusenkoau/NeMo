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
from nemo.collections.asr.parts.rnnt_triton import ConsistencyRNNTLoss, consistency_rnnt_kld


def get_devices_for_testing(use_cpu_always: bool = False) -> list[torch.device]:
    devices = [torch.device("cpu")] if use_cpu_always else []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    if torch.mps.is_available():
        devices.append(torch.device("mps"))

    if len(devices) == 0:
        # no fast device for testing, add CPU
        devices.append(torch.device("cpu"))
    return devices


DEVICES = get_devices_for_testing(use_cpu_always=False)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_single_frame_single_token(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Edge case: minimal tensor sizes (T=1, U=1)."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(1, 1, 2, 3, device=device)  # [B=1, T=1, U+1=2, V=3]
    student_logits = torch.randn(1, 1, 2, 3, device=device)
    targets = torch.randint(0, 2, (1, 1), device=device)  # blank_id=2

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=2,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_identity_kl_zero(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """When teacher == student, KL should be 0."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 3, 5, device=device)  # [B, T, U+1, V]
    targets = torch.randint(0, 4, (2, 2), device=device)  # [B, U], exclude blank_id=4

    loss = consistency_rnnt_kld(
        teacher_logits=logits,
        student_logits=logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_non_negativity(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """KL divergence should always be >= 0."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss >= 0


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_symmetrical_swap_invariance(device: torch.device, use_blank: bool, reduction: str):
    """With symmetrical=True, swapping teacher/student gives same loss."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss1 = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=True,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss2 = consistency_rnnt_kld(
        teacher_logits=student_logits,
        student_logits=teacher_logits,
        targets=targets,
        blank_id=4,
        symmetrical=True,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert torch.isclose(loss1, loss2, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_non_symmetrical_different_on_swap(device: torch.device, use_blank: bool, reduction: str):
    """With symmetrical=False, swap gives different loss."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss1 = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=False,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss2 = consistency_rnnt_kld(
        teacher_logits=student_logits,
        student_logits=teacher_logits,
        targets=targets,
        blank_id=4,
        symmetrical=False,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert not torch.isclose(loss1, loss2, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_gradient_flow(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Verify gradients flow to student logits."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device, requires_grad=True)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss.backward()

    assert student_logits.grad is not None
    assert not torch.all(student_logits.grad == 0)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_gradient_numerical_check(device: torch.device, use_blank: bool, reduction: str):
    """Numerical gradient verification.

    Note: Only testing symmetrical=False because symmetrical=True uses detach()
    on the student logprobs in the reverse direction, which causes gradcheck to fail
    (detach prevents gradient flow, making numerical and analytical gradients differ).
    """
    if device.type == "mps":
        pytest.skip("MPS does not support float64")
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 3, 2, 4, dtype=torch.float64, device=device)
    student_logits = torch.randn(2, 3, 2, 4, dtype=torch.float64, device=device, requires_grad=True)
    targets = torch.randint(0, 3, (2, 1), device=device)  # exclude blank_id=3

    def loss_fn(s_logits):
        return consistency_rnnt_kld(
            teacher_logits=teacher_logits,
            student_logits=s_logits,
            targets=targets,
            blank_id=3,
            symmetrical=False,
            use_blank=use_blank,
            reduction=reduction,
        )

    assert torch.autograd.gradcheck(loss_fn, (student_logits,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_variable_lengths(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Test with different src_lengths/tgt_lengths."""
    torch.manual_seed(42)
    batch_size, T, U_plus_1, V = 3, 5, 4, 6
    teacher_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    student_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    targets = torch.randint(0, V - 1, (batch_size, U_plus_1 - 1), device=device)  # blank_id = V-1
    src_lengths = torch.tensor([5, 3, 4], device=device)
    tgt_lengths = torch.tensor([3, 2, 1], device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=V - 1,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_masking_correctness(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Verify padded positions are ignored by checking that loss is same
    regardless of values in padded regions."""
    torch.manual_seed(42)
    batch_size, T, U_plus_1, V = 2, 5, 4, 6
    teacher_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    student_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    targets = torch.randint(0, V - 1, (batch_size, U_plus_1 - 1), device=device)
    src_lengths = torch.tensor([3, 2], device=device)
    tgt_lengths = torch.tensor([2, 1], device=device)

    loss1 = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=V - 1,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )

    # Modify values in padded regions with deterministic values
    teacher_logits_modified = teacher_logits.clone()
    student_logits_modified = student_logits.clone()
    # Padded time frames for sample 0: t >= 3
    teacher_logits_modified[0, 3:, :, :] = 100.0
    student_logits_modified[0, 3:, :, :] = -100.0
    # Padded time frames for sample 1: t >= 2
    teacher_logits_modified[1, 2:, :, :] = 100.0
    student_logits_modified[1, 2:, :, :] = -100.0

    loss2 = consistency_rnnt_kld(
        teacher_logits=teacher_logits_modified,
        student_logits=student_logits_modified,
        targets=targets,
        blank_id=V - 1,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )

    assert torch.isclose(loss1, loss2, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
def test_reduction_modes_consistency(device: torch.device, symmetrical: bool, use_blank: bool):
    """Both reductions give valid (non-negative, finite) results."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss_mean = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction="mean",
    )
    loss_mean_volume = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction="mean_volume",
    )

    assert loss_mean >= 0
    assert loss_mean_volume >= 0
    assert torch.isfinite(loss_mean)
    assert torch.isfinite(loss_mean_volume)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_batch_size_one(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Edge case: batch_size = 1."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(1, 4, 3, 5, device=device)
    student_logits = torch.randn(1, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (1, 2), device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_large_vocab(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Edge case: large vocabulary."""
    torch.manual_seed(42)
    V = 1024
    teacher_logits = torch.randn(2, 4, 3, V, device=device)
    student_logits = torch.randn(2, 4, 3, V, device=device)
    targets = torch.randint(0, V - 1, (2, 2), device=device)  # blank_id = V-1

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=V - 1,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_module_api(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Test nn.Module wrapper."""
    torch.manual_seed(42)
    module = ConsistencyRNNTLoss(
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )

    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = module(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)
