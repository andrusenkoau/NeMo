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

from typing import Dict, Optional

import torch
from torch import Tensor

from nemo.utils import logging

__all__ = ['LangPromptMixin']


class LangPromptMixin:
    """Language/task prompt conditioning for encoder-decoder ASR models ("unified" models).

    A one-hot language/task vector is concatenated to every encoder output frame and projected back
    to the encoder dimension, letting the decoder condition on the requested language. The mixin owns
    the projection module, the prompt vocabulary, and the prompt tensor construction; host models only
    need to call :meth:`setup_lang_prompt` during ``__init__`` and :meth:`apply_lang_prompt` (or
    :meth:`apply_lang_prompt_for_transcribe`) after the encoder.

    Enabled by the following model config, which must be present in the checkpoint:

    .. code-block:: yaml

        model:
          model_defaults:
            initialize_prompt_feature: true
            num_prompts: 128
            prompt_dictionary: {en-US: 0, de-DE: 9, ..., unk: 127}

    Models that were not trained with prompt conditioning keep ``use_prompt = False`` and are entirely
    unaffected.
    """

    # Plain class-level defaults so hosts can read these before (or without) calling
    # ``setup_lang_prompt``. ``prompt_kernel`` is intentionally NOT declared here: it is an
    # ``nn.Module`` and a class attribute would shadow ``nn.Module.__getattr__``'s lookup into
    # ``_modules`` once the real module is registered.
    use_prompt: bool = False
    num_prompts: Optional[int] = None
    prompt_dictionary: Optional[Dict[str, int]] = None

    def setup_lang_prompt(self) -> None:
        """Build the prompt projection if the model config asks for it, otherwise do nothing.

        Safe to call unconditionally from a host model's ``__init__``.
        """
        # Models that mix in their own prompt machinery (``PromptStreamingMixin``, used by the
        # prompt-aware training classes) build and own ``prompt_kernel`` themselves.
        if hasattr(self, 'initialize_prompt_feature'):
            return

        model_defaults = self.cfg.get('model_defaults') or {}
        if not model_defaults.get('initialize_prompt_feature', False):
            return

        prompt_dictionary = model_defaults.get('prompt_dictionary', None)
        if not prompt_dictionary:
            raise ValueError(
                "`model_defaults.prompt_dictionary` must be a non-empty mapping of language/task name "
                "to prompt index when `model_defaults.initialize_prompt_feature=true`."
            )

        enc_hidden = model_defaults.get('enc_hidden', None)
        if enc_hidden is None:
            raise ValueError(
                "`model_defaults.enc_hidden` is required to size the prompt projection when "
                "`model_defaults.initialize_prompt_feature=true`."
            )

        self.use_prompt = True
        self.num_prompts = int(model_defaults.get('num_prompts', 128))
        self.prompt_dictionary = prompt_dictionary
        self.prompt_kernel = _Float32PromptProjection(
            torch.nn.Linear(self.num_prompts + enc_hidden, enc_hidden * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_hidden * 2, enc_hidden),
        )
        logging.info(
            f"Prompt conditioning enabled: num_prompts={self.num_prompts}, "
            f"languages={list(self.prompt_dictionary.keys())}"
        )

    def resolve_prompt_id(self, target_lang: Optional[str]) -> int:
        """Resolve a language/task name to its prompt index.

        The prompt projection is part of the trained forward pass and must always be applied, so this
        always returns an index. When ``target_lang`` is missing or unknown, the model's
        language-agnostic ``unk`` prompt is used and a warning is logged.

        Args:
            target_lang: A key of ``prompt_dictionary`` (e.g. ``"en-US"``), or None.

        Returns:
            The prompt index to condition on.
        """
        if not self.use_prompt:
            raise ValueError(f"{type(self).__name__} was not trained with prompt conditioning.")

        if target_lang is not None and target_lang in self.prompt_dictionary:
            return self.prompt_dictionary[target_lang]

        available = list(self.prompt_dictionary.keys())
        preview = f"{available[:10]}{'...' if len(available) > 10 else ''}"
        if target_lang is None:
            logging.warning(
                "No `target_lang` provided for prompt-conditioned model; falling back to the 'unk' prompt. "
                "Pass `target_lang=<lang>` (e.g. target_lang=en-US) to force a specific language."
            )
        else:
            logging.warning(
                f"Unknown target_lang='{target_lang}' (available: {preview}); falling back to the 'unk' prompt."
            )

        unk_id = self.prompt_dictionary.get('unk')
        if unk_id is None:
            raise ValueError(
                "No 'unk' entry in the model's prompt_dictionary to use as a fallback prompt. "
                f"Please pass an explicit `target_lang`. Available: {preview}"
            )
        return unk_id

    def create_onehot_prompt(
        self, batch_size: int, prompt_id: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """Create a one-hot language/task prompt shared by a whole batch.

        Args:
            batch_size: Number of utterances in the batch.
            prompt_id: Prompt index, e.g. from :meth:`resolve_prompt_id`.
            dtype: Dtype of the returned tensor.
            device: Device of the returned tensor.

        Returns:
            One-hot prompt of shape ``(batch_size, num_prompts)``.
        """
        prompt = torch.zeros(batch_size, self.num_prompts, dtype=dtype, device=device)
        prompt[:, prompt_id] = 1.0
        return prompt

    def apply_lang_prompt(self, encoded: Tensor, prompt: Tensor) -> Tensor:
        """Condition the encoder output on a language/task prompt.

        Args:
            encoded: Encoder output of shape ``(B, D, T)``.
            prompt: One-hot prompt of shape ``(B, num_prompts)``, broadcast across time; or a
                per-frame prompt of shape ``(B, T, num_prompts)`` whose ``T`` must match ``encoded``.

        Returns:
            Prompt-conditioned encoder output of shape ``(B, D, T)``.
        """
        if not self.use_prompt:
            raise ValueError(
                f"A prompt was passed to {type(self).__name__}, which was not trained with prompt conditioning."
            )

        encoded = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        out_dtype = encoded.dtype
        batch_size, time_steps, _ = encoded.shape

        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(1).expand(-1, time_steps, -1)
        elif prompt.dim() == 3:
            # Reject a mismatch rather than truncating or zero-padding: a zero-padded frame is a prompt
            # the model never saw in training, and silently mis-conditions the tail of every chunk.
            if prompt.shape[1] != time_steps:
                raise ValueError(
                    f"prompt has {prompt.shape[1]} time steps but the encoder produced {time_steps}. "
                    "Pass a (B, num_prompts) prompt to broadcast across time instead of precomputing "
                    "the time dimension."
                )
        else:
            raise ValueError(f"Expected a 2D or 3D prompt, got shape {tuple(prompt.shape)}.")

        if prompt.shape[0] != batch_size or prompt.shape[-1] != self.num_prompts:
            raise ValueError(
                f"Expected a prompt with batch size {batch_size} and {self.num_prompts} classes, "
                f"got shape {tuple(prompt.shape)}."
            )

        # The projection runs in float32 (see ``_Float32PromptProjection``).
        with torch.amp.autocast(device_type=encoded.device.type, enabled=False):
            encoded = self.prompt_kernel(torch.cat([encoded.float(), prompt.float()], dim=-1))
        return encoded.to(out_dtype).transpose(1, 2)  # (B, T, D) -> (B, D, T)

    def apply_lang_prompt_for_transcribe(self, encoded: Tensor, target_lang: Optional[str]) -> Tensor:
        """Resolve ``target_lang`` and condition the encoder output on it, for whole-batch inference.

        A no-op for models without prompt conditioning, so transcription paths can call it directly.

        Args:
            encoded: Encoder output of shape ``(B, D, T)``.
            target_lang: A key of ``prompt_dictionary``, or None to use the ``unk`` prompt.

        Returns:
            Encoder output of shape ``(B, D, T)``, conditioned if the model supports prompts.
        """
        if not self.use_prompt:
            return encoded

        prompt_id = self.resolve_prompt_id(target_lang)
        prompt = self.create_onehot_prompt(encoded.shape[0], prompt_id, dtype=encoded.dtype, device=encoded.device)
        return self.apply_lang_prompt(encoded, prompt)


class _Float32PromptProjection(torch.nn.Sequential):
    """Prompt projection that stays in float32 through model-wide dtype casts.

    The one-hot prompt contributes a single unit-magnitude feature alongside ``enc_hidden`` encoder
    activations of far larger magnitude. Measured on a released 0.6B unified checkpoint, the
    language-discriminative part of the output is 0.1-1% of its total magnitude, so in bfloat16 it
    falls into the mantissa noise and the projection becomes effectively language-independent while
    decoding silently ignores the requested language. Keeping just this small MLP in float32 restores
    conditioning at negligible cost.

    Device moves are honoured as usual; only the floating point dtype is pinned. Note that the
    dtype change is dropped rather than undone: casting down and back up would already have
    discarded the low-order mantissa bits this class exists to protect.
    """

    def _apply(self, fn, recurse: bool = True):
        def keep_float32(tensor: Tensor) -> Tensor:
            transformed = fn(tensor)
            if tensor.is_floating_point() and transformed.dtype is not torch.float32:
                # Follow the device (and any other placement change) but keep full precision by
                # re-deriving from the original tensor instead of the already-downcast result.
                return tensor.to(device=transformed.device, dtype=torch.float32)
            return transformed

        return super()._apply(keep_float32, recurse=recurse)
