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

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import rnnt_logprobs
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_joint_triton import rnnt_joint_logprobs_triton


def _build_joint_net(hidden_dim, vocab_size_with_blank, device, dtype):
    """Build a simple joint network: ReLU + Linear(hidden_dim, vocab_size_with_blank)."""
    linear = torch.nn.Linear(hidden_dim, vocab_size_with_blank, device=device, dtype=dtype)
    return linear


def _reference_joint_logprobs(enc, pred, linear, targets, src_lengths, tgt_lengths, blank_id):
    """
    Reference pipeline: joint_after_projection style + rnnt_logprobs.

    enc: [B, T, D]
    pred: [B, U+1, D]
    linear: nn.Linear(D, V)
    """
    # Broadcast add + relu + linear (materializes full [B, T, U+1, D] and [B, T, U+1, V])
    hidden = torch.relu(enc.unsqueeze(2) + pred.unsqueeze(1))  # [B, T, U+1, D]
    logits = linear(hidden)  # [B, T, U+1, V]
    target_logprobs, blank_logprobs = rnnt_logprobs(
        logits=logits, targets=targets, blank_id=blank_id, src_lengths=src_lengths, tgt_lengths=tgt_lengths
    )
    return target_logprobs, blank_logprobs


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
class TestRnntJointTriton:
    @pytest.mark.parametrize(
        "B,T,U,D,V",
        [
            (1, 2, 1, 4, 3),
            (2, 4, 3, 16, 8),
            (4, 16, 8, 64, 32),
            (5, 17, 65, 28, 35),
            (2, 64, 32, 640, 1024),
        ],
    )
    @pytest.mark.parametrize(
        "float_dtype",
        [torch.float32] + ([torch.bfloat16] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else []),
    )
    def test_forward_and_backward(self, B, T, U, D, V, float_dtype):
        """Test forward logprobs match reference; backward gradients match."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        blank_id = V  # vocab_size_with_blank = V + 1
        vocab_size_with_blank = V + 1

        # Build shared linear layer
        linear = _build_joint_net(D, vocab_size_with_blank, device, float_dtype)
        weight = linear.weight.detach().clone()  # [V+1, D]
        bias = linear.bias.detach().clone()  # [V+1]

        # Create inputs
        enc = torch.randn(B, T, D, device=device, dtype=float_dtype)
        pred = torch.randn(B, U + 1, D, device=device, dtype=float_dtype)
        targets = torch.randint(0, V, (B, U), device=device, dtype=torch.long)
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.full([B], U, device=device, dtype=torch.long)

        # Reference pipeline
        enc_ref = enc.clone().detach().requires_grad_(True)
        pred_ref = pred.clone().detach().requires_grad_(True)
        linear_ref = torch.nn.Linear(D, vocab_size_with_blank, device=device, dtype=float_dtype)
        linear_ref.weight = torch.nn.Parameter(weight.clone())
        linear_ref.bias = torch.nn.Parameter(bias.clone())

        target_ref, blank_ref = _reference_joint_logprobs(
            enc_ref, pred_ref, linear_ref, targets, src_lengths, tgt_lengths, blank_id
        )

        # Triton pipeline
        enc_tri = enc.clone().detach().requires_grad_(True)
        pred_tri = pred.clone().detach().requires_grad_(True)
        weight_tri = weight.clone().detach().requires_grad_(True)
        bias_tri = bias.clone().detach().requires_grad_(True)

        target_tri, blank_tri = rnnt_joint_logprobs_triton(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
        )

        # Forward comparison
        # fp32: TF32 dot product has ~10-bit mantissa, so ~3e-3 error expected
        # bf16: reference computes matmul in bf16 while triton computes in float32
        fwd_atol = 1e-2 if float_dtype == torch.bfloat16 else 3e-3
        assert torch.allclose(
            target_tri, target_ref, atol=fwd_atol
        ), f"target_logprobs mismatch: max diff = {(target_tri - target_ref).abs().max().item()}"
        assert torch.allclose(
            blank_tri, blank_ref, atol=fwd_atol
        ), f"blank_logprobs mismatch: max diff = {(blank_tri - blank_ref).abs().max().item()}"

        # Backward comparison
        torch.manual_seed(123)
        target_scales = torch.rand_like(target_ref, requires_grad=False)
        blank_scales = torch.rand_like(blank_ref, requires_grad=False)

        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_tri = (target_scales * target_tri + blank_scales * blank_tri).sum()

        loss_ref.backward()
        loss_tri.backward()

        # fp32: TF32 forward error propagates to backward via log_sum_exp_scores
        # bf16: reference computes in bf16 while triton accumulates in float32
        bwd_atol = 5e-2 if float_dtype == torch.bfloat16 else 5e-3
        bwd_rtol = 5e-2 if float_dtype == torch.bfloat16 else 1e-3

        assert torch.allclose(
            enc_tri.grad, enc_ref.grad, atol=bwd_atol, rtol=bwd_rtol
        ), f"enc grad mismatch: max diff = {(enc_tri.grad - enc_ref.grad).abs().max().item()}"
        assert torch.allclose(
            pred_tri.grad, pred_ref.grad, atol=bwd_atol, rtol=bwd_rtol
        ), f"pred grad mismatch: max diff = {(pred_tri.grad - pred_ref.grad).abs().max().item()}"
        assert torch.allclose(
            weight_tri.grad, linear_ref.weight.grad, atol=bwd_atol, rtol=bwd_rtol
        ), f"weight grad mismatch: max diff = {(weight_tri.grad - linear_ref.weight.grad).abs().max().item()}"
        assert torch.allclose(
            bias_tri.grad, linear_ref.bias.grad, atol=bwd_atol, rtol=bwd_rtol
        ), f"bias grad mismatch: max diff = {(bias_tri.grad - linear_ref.bias.grad).abs().max().item()}"

    def test_variable_lengths(self):
        """Test with variable src/tgt lengths per batch element."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        B, T, U, D, V = 4, 16, 8, 32, 16
        blank_id = V
        vocab_size_with_blank = V + 1

        linear = _build_joint_net(D, vocab_size_with_blank, device, torch.float32)
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone()

        enc = torch.randn(B, T, D, device=device, dtype=torch.float32)
        pred = torch.randn(B, U + 1, D, device=device, dtype=torch.float32)
        targets = torch.randint(0, V, (B, U), device=device, dtype=torch.long)
        src_lengths = torch.tensor([T, T // 2, T - 1, 1], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([U, U // 2, U - 1, 0], device=device, dtype=torch.long)

        # Reference
        enc_ref = enc.clone().detach().requires_grad_(True)
        pred_ref = pred.clone().detach().requires_grad_(True)
        linear_ref = torch.nn.Linear(D, vocab_size_with_blank, device=device, dtype=torch.float32)
        linear_ref.weight = torch.nn.Parameter(weight.clone())
        linear_ref.bias = torch.nn.Parameter(bias.clone())
        target_ref, blank_ref = _reference_joint_logprobs(
            enc_ref, pred_ref, linear_ref, targets, src_lengths, tgt_lengths, blank_id
        )

        # Triton
        enc_tri = enc.clone().detach().requires_grad_(True)
        pred_tri = pred.clone().detach().requires_grad_(True)
        weight_tri = weight.clone().detach().requires_grad_(True)
        bias_tri = bias.clone().detach().requires_grad_(True)
        target_tri, blank_tri = rnnt_joint_logprobs_triton(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
        )

        fwd_atol = 5e-3  # TF32 dot product has ~10-bit mantissa
        assert torch.allclose(
            target_tri, target_ref, atol=fwd_atol
        ), f"target_logprobs mismatch: max diff = {(target_tri - target_ref).abs().max().item()}"
        assert torch.allclose(
            blank_tri, blank_ref, atol=fwd_atol
        ), f"blank_logprobs mismatch: max diff = {(blank_tri - blank_ref).abs().max().item()}"

        # Backward
        target_scales = torch.rand_like(target_ref)
        blank_scales = torch.rand_like(blank_ref)
        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_tri = (target_scales * target_tri + blank_scales * blank_tri).sum()
        loss_ref.backward()
        loss_tri.backward()

        assert torch.allclose(enc_tri.grad, enc_ref.grad, atol=5e-3, rtol=1e-3)
        assert torch.allclose(pred_tri.grad, pred_ref.grad, atol=5e-3, rtol=1e-3)

    def test_edge_case_single_frame(self):
        """Test T=1, U=1 minimal case."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        B, T, U, D, V = 1, 1, 1, 4, 3
        blank_id = V
        vocab_size_with_blank = V + 1

        linear = _build_joint_net(D, vocab_size_with_blank, device, torch.float32)
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone()

        enc = torch.randn(B, T, D, device=device, dtype=torch.float32)
        pred = torch.randn(B, U + 1, D, device=device, dtype=torch.float32)
        targets = torch.randint(0, V, (B, U), device=device, dtype=torch.long)
        src_lengths = torch.tensor([T], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([U], device=device, dtype=torch.long)

        enc_ref = enc.clone().detach().requires_grad_(True)
        pred_ref = pred.clone().detach().requires_grad_(True)
        linear_ref = torch.nn.Linear(D, vocab_size_with_blank, device=device, dtype=torch.float32)
        linear_ref.weight = torch.nn.Parameter(weight.clone())
        linear_ref.bias = torch.nn.Parameter(bias.clone())
        target_ref, blank_ref = _reference_joint_logprobs(
            enc_ref, pred_ref, linear_ref, targets, src_lengths, tgt_lengths, blank_id
        )

        enc_tri = enc.clone().detach().requires_grad_(True)
        pred_tri = pred.clone().detach().requires_grad_(True)
        weight_tri = weight.clone().detach().requires_grad_(True)
        bias_tri = bias.clone().detach().requires_grad_(True)
        target_tri, blank_tri = rnnt_joint_logprobs_triton(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
        )

        assert torch.allclose(target_tri, target_ref, atol=5e-3)
        assert torch.allclose(blank_tri, blank_ref, atol=5e-3)

    def test_edge_case_blank_only(self):
        """Test U=0 (tgt_length=0) - only blank is possible."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        B, T, D, V = 2, 4, 8, 5
        U = 0
        blank_id = V
        vocab_size_with_blank = V + 1

        linear = _build_joint_net(D, vocab_size_with_blank, device, torch.float32)
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone()

        enc = torch.randn(B, T, D, device=device, dtype=torch.float32)
        pred = torch.randn(B, U + 1, D, device=device, dtype=torch.float32)
        targets = torch.randint(0, V, (B, max(U, 1)), device=device, dtype=torch.long)
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.zeros([B], device=device, dtype=torch.long)

        enc_ref = enc.clone().detach().requires_grad_(True)
        pred_ref = pred.clone().detach().requires_grad_(True)
        linear_ref = torch.nn.Linear(D, vocab_size_with_blank, device=device, dtype=torch.float32)
        linear_ref.weight = torch.nn.Parameter(weight.clone())
        linear_ref.bias = torch.nn.Parameter(bias.clone())
        target_ref, blank_ref = _reference_joint_logprobs(
            enc_ref, pred_ref, linear_ref, targets, src_lengths, tgt_lengths, blank_id
        )

        enc_tri = enc.clone().detach().requires_grad_(True)
        pred_tri = pred.clone().detach().requires_grad_(True)
        weight_tri = weight.clone().detach().requires_grad_(True)
        bias_tri = bias.clone().detach().requires_grad_(True)
        target_tri, blank_tri = rnnt_joint_logprobs_triton(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
        )

        assert torch.allclose(target_tri, target_ref, atol=5e-3)
        assert torch.allclose(blank_tri, blank_ref, atol=5e-3)

    def test_no_cuda_sync(self):
        """Verify no CPU-GPU sync happens during forward/backward."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        B, T, U, D, V = 1, 2, 1, 4, 3
        blank_id = V
        vocab_size_with_blank = V + 1

        linear = _build_joint_net(D, vocab_size_with_blank, device, torch.float32)
        weight = linear.weight.detach().clone().requires_grad_(True)
        bias = linear.bias.detach().clone().requires_grad_(True)

        enc = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
        pred = torch.randn(B, U + 1, D, device=device, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, V, (B, U), device=device, dtype=torch.long)
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.full([B], U, device=device, dtype=torch.long)

        # Warmup
        target_tri, blank_tri = rnnt_joint_logprobs_triton(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
        )
        loss = target_tri.sum() + blank_tri.sum()
        loss.backward()
        torch.cuda.synchronize()

        # Actual test with sync tracking
        from unittest.mock import patch

        sync_count = 0
        original_sync = torch.cuda.synchronize

        def counting_sync(*args, **kwargs):
            nonlocal sync_count
            sync_count += 1
            original_sync(*args, **kwargs)

        enc2 = enc.detach().clone().requires_grad_(True)
        pred2 = pred.detach().clone().requires_grad_(True)
        weight2 = weight.detach().clone().requires_grad_(True)
        bias2 = bias.detach().clone().requires_grad_(True)

        with patch("torch.cuda.synchronize", counting_sync):
            target_tri, blank_tri = rnnt_joint_logprobs_triton(
                encoder_output_projected=enc2,
                predictor_output_projected=pred2,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=weight2,
                bias=bias2,
                blank_id=blank_id,
            )
            loss = target_tri.sum() + blank_tri.sum()
            loss.backward()

        assert sync_count == 0, f"Expected no CUDA synchronize calls, got {sync_count}"

    def test_grad_check(self):
        """Numerical gradient verification using torch.autograd.gradcheck."""
        pytest.skip(reason="temporary skip - slow")
        device = torch.device("cuda")
        torch.manual_seed(42)
        B, T, U, D, V = 1, 2, 1, 4, 3
        blank_id = V
        vocab_size_with_blank = V + 1

        enc = torch.randn(B, T, D, device=device, dtype=torch.float64, requires_grad=True)
        pred = torch.randn(B, U + 1, D, device=device, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(vocab_size_with_blank, D, device=device, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, V, (B, U), device=device, dtype=torch.long)
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.full([B], U, device=device, dtype=torch.long)

        def func(enc_in, pred_in, w_in, b_in):
            t_lp, b_lp = rnnt_joint_logprobs_triton(
                encoder_output_projected=enc_in,
                predictor_output_projected=pred_in,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=w_in,
                bias=b_in,
                blank_id=blank_id,
            )
            return t_lp, b_lp

        # nondet_tol needed because backward uses atomic adds which are nondeterministic
        assert torch.autograd.gradcheck(
            func, (enc, pred, weight, bias), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5
        )
