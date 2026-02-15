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


from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_joint_triton import rnnt_joint_logprobs_triton


# tests: instantiate joint (currently without dropout - use .eval()), and only "relu" activation, generate encoder_output_projected and predictor_output_projected
# compare:
# (1) get logprobs for RNN-T loss (current pipeline): get logits from joint, get rnnt logprobs using nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs.rnnt_logprobs
# (2) get efficient logprobs: use rnnt_joint_logprobs_triton
# compare logprobs and gradients (using sum with random weights), no need to use RNN-T loss here
