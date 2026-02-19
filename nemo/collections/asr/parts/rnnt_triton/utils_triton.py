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

import triton
import triton.language as tl


@triton.jit
def log_add_exp(log_probs_1, log_probs_2):
    max_score = tl.maximum(log_probs_1, log_probs_2)
    return max_score + tl.log(tl.exp(log_probs_1 - max_score) + tl.exp(log_probs_2 - max_score))


@triton.jit
def add_at_range_(x: tl.tensor, y: tl.tensor, start, axis):
    # TODO: optimize
    # TODO: add tests
    x_offsets = tl.arange(0, x.shape[axis])
    y_len = y.shape[axis]
    num_axes = len(x.shape)
    mask = (x_offsets >= start) & (x_offsets < y_len)
    y_indices_safe_to_x = tl.where(mask, x_offsets - start, 0)
    broadcastable_shape = [1] * axis + [y_len] + [1] * (num_axes - axis)
    y_to_x_expanded = y.gather(y_indices_safe_to_x.reshape(broadcastable_shape).broadcast_to(x.shape), axis=axis)
    x += tl.where(mask.reshape(broadcastable_shape), y_to_x_expanded, 0)
