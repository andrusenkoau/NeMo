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
def sum_at_range(x: tl.tensor, y: tl.tensor, start, axis: tl.constexpr):
    """
    Return x[..., 0:start] + (x[..., start:start+y.shape[axis]] + y) + x[.., start+y.shape[axis]:]
    """
    # TODO: add tests
    # TODO: optimize (?)
    x_offsets = tl.arange(0, x.shape[axis])
    mask = (x_offsets >= start) & (x_offsets < start + y.shape[axis])
    y_indices_safe_to_x = tl.where(mask, x_offsets - start, 0)
    broadcastable_shape: tl.constexpr = [1] * axis + [x.shape[axis]] + ([1] * (len(x.shape) - axis - 1))
    y_to_x_expanded = y.gather(y_indices_safe_to_x.reshape(broadcastable_shape).broadcast_to(x.shape), axis=axis)
    y_to_x_expanded = tl.where(mask.reshape(broadcastable_shape), y_to_x_expanded, 0.0)
    return x + y_to_x_expanded
