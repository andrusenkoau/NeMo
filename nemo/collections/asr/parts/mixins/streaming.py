# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

import torch


class StreamingEncoder(ABC):
    @abstractmethod
    def setup_streaming_params(
        self,
        max_look_ahead: int = 10000,
    ):
        """
        This function sets the needed values and parameters to perform streaming. The configuration (CacheAwareStreamingConfig) need to be stored in self.streaming_cfg.
        The streaming configuration is needed to simulate streaming inference. It would set the following
        """
        pass

    @abstractmethod
    def get_initial_cache_state(self, batch_size, dtype, device, max_dim):
        pass

    @staticmethod
    def to_numpy(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def cache_aware_stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        bypass_pre_encode=False,
    ):
        if self.streaming_cfg is None:
            self.setup_streaming_params()
        if drop_extra_pre_encoded is not None:
            prev_drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded
            self.streaming_cfg.drop_extra_pre_encoded = drop_extra_pre_encoded
        else:
            prev_drop_extra_pre_encoded = None

        if processed_signal_length is None:
            processed_signal_length = processed_signal.new_full(processed_signal.size(0), processed_signal.size(-1))

        encoder_output = self(
            audio_signal=processed_signal,
            length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

        encoder_output = self.streaming_post_process(encoder_output, keep_all_outputs=keep_all_outputs)

        if prev_drop_extra_pre_encoded is not None:
            self.streaming_cfg.drop_extra_pre_encoded = prev_drop_extra_pre_encoded

        return encoder_output

    def cache_aware_stream_step_v2(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
    ):
        """Cache-aware stream step with incrementally growing attention cache.

        Unlike :meth:`cache_aware_stream_step`, the attention cache starts empty
        and grows by ``cache_keep_size`` frames per step until reaching
        ``last_channel_cache_size``.  The convolution cache is always full-size
        (initialized to zeros on the first call, matching offline zero-padding).

        On the first call (``cache_last_channel=None``), the encoder processes
        the full input with a zero-size attention cache — equivalent to the
        offline forward path.  The cache is extracted naturally from the layer
        outputs.  No double-encoding is needed.
        """
        if self.streaming_cfg is None:
            self.setup_streaming_params()

        if drop_extra_pre_encoded is not None:
            prev_drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded
            self.streaming_cfg.drop_extra_pre_encoded = drop_extra_pre_encoded
        else:
            prev_drop_extra_pre_encoded = None

        if processed_signal_length is None:
            processed_signal_length = processed_signal.new_full(
                (processed_signal.size(0),), processed_signal.size(-1), dtype=torch.int64
            )

        is_first_step = cache_last_channel is None

        if is_first_step:
            batch_size = processed_signal.size(0)
            device = processed_signal.device
            dtype = processed_signal.dtype
            cache_last_channel = torch.zeros(
                len(self.layers), batch_size, 0, self.d_model,
                device=device, dtype=dtype,
            )
            last_time_cache_size = self.conv_context_size[0]
            cache_last_time = torch.zeros(
                len(self.layers), batch_size, self.d_model, last_time_cache_size,
                device=device, dtype=dtype,
            )
            cache_last_channel_len = torch.zeros(
                batch_size, device=device, dtype=torch.int64,
            )

        self.update_max_seq_length(seq_length=processed_signal.size(-1), device=processed_signal.device)
        encoder_output = self.forward_internal(
            audio_signal=processed_signal,
            length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            dynamic_cache=True,
        )

        (encoded, encoded_len, cache_ch_next, cache_t_next, cache_ch_len_next) = encoder_output

        # Trim attention cache to the configured maximum.
        max_cache = self.streaming_cfg.last_channel_cache_size
        if cache_ch_next is not None and cache_ch_next.size(2) > max_cache:
            cache_ch_next = cache_ch_next[:, :, -max_cache:, :]
        if cache_ch_next is not None:
            cache_ch_len_next = torch.full_like(
                cache_ch_len_next, fill_value=cache_ch_next.size(2)
            )

        # Truncate encoder output to valid_out_len for non-first, non-last steps.
        # First step returns ALL frames so the caller can strip left context.
        if (
            not is_first_step
            and self.streaming_cfg.valid_out_len > 0
            and (not keep_all_outputs or self.att_context_style == "regular")
        ):
            encoded = encoded[:, :, : self.streaming_cfg.valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=self.streaming_cfg.valid_out_len)

        if prev_drop_extra_pre_encoded is not None:
            self.streaming_cfg.drop_extra_pre_encoded = prev_drop_extra_pre_encoded

        return (encoded, encoded_len, cache_ch_next, cache_t_next, cache_ch_len_next)
