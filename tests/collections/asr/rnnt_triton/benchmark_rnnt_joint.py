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

from nemo.collections.asr.modules import RNNTJoint
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.core.classes.common import typecheck
import torch
from nemo.collections.asr.parts.rnnt_triton.rnnt_joint_triton import rnnt_joint_logprobs_triton
from nemo.collections.asr.parts.rnnt_triton.rnnt_loss_triton import rnnt_loss_from_logprobs_triton


def logic_explanation():
    # this function is to explain the logic
    # delete it after implementing benchmark

    typecheck.set_typecheck_enabled(False)  # globally disable typechecks
    device = torch.device("cuda")
    vocab_size = 1024
    float_dtype = torch.float32  # we will use float32 and bfloat16

    # instantiate joint
    joint = RNNTJoint(
        jointnet={
            "joint_hidden": 640,
            "encoder_hidden": 512,
            "pred_hidden": 640,
            "activation": "relu",
            "dropout": 0.2,
        },
        num_classes=1024,
    ).to(device)

    joint.eval()  # disable currently dropout
    # TODO: support dropout later
    loss = RNNTLoss(num_classes=vocab_size, loss_name="rnnt_triton", reduction=None)  # rnnt_triton or warprnnt_numba

    batch_size, src_max_length, tgt_max_length = 64, 150, 37
    encoder_output = torch.rand([batch_size, src_max_length, 512], device=device, dtype=float_dtype)
    predictor_output = torch.rand([batch_size, tgt_max_length + 1, 640], device=device, dtype=float_dtype)
    targets = torch.randint(0, vocab_size - 1, [batch_size, tgt_max_length], dtype=torch.long, device=device)
    tgt_lengths = torch.full([batch_size], fill_value=tgt_max_length, dtype=torch.long, device=device)
    src_lengths = torch.full([batch_size], fill_value=src_max_length, dtype=torch.long, device=device)

    encoder_output = joint.project_encoder(encoder_output)
    predictor_output = joint.project_prednet(predictor_output)

    # we want to optimize the following
    # regular joint: get logits, calculate loss
    logits = joint.joint_after_projection(f=encoder_output, g=predictor_output)
    loss = loss(
        log_probs=logits,
        targets=targets,
        input_lengths=src_lengths,
        target_lengths=tgt_lengths,
    ).mean()
    print(loss)

    # new pipeline: get (efficiently logprobs), calculate loss
    target_logprobs, blank_logprobs = rnnt_joint_logprobs_triton(
        encoder_output_projected=encoder_output,
        predictor_output_projected=predictor_output,
        targets=targets,
        tgt_lengths=tgt_lengths,
        src_lengths=src_lengths,
        weight=joint.joint_net[2].weight,
        bias=joint.joint_net[2].bias,
        blank_id=vocab_size,
        activation="relu",
    )
    loss2 = rnnt_loss_from_logprobs_triton(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    ).mean()
    print(loss2)
