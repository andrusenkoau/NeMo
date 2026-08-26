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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecRNNTModel
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__ as NUMBA_MINIMUM_VERSION
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE, numba_cpu_is_supported

NUMBA_RNNT_LOSS_AVAILABLE = numba_cpu_is_supported(NUMBA_MINIMUM_VERSION)

ENC_HIDDEN = 64
NUM_PROMPTS = 8
PROMPT_DICTIONARY = {'en-US': 0, 'en': 0, 'de-DE': 3, 'ja-JP': 5, 'unk': 7}

requires_numba = pytest.mark.skipif(
    not NUMBA_RNNT_LOSS_AVAILABLE,
    reason=f'RNNTLoss has not been compiled with appropriate numba version. {NUMBA_INSTALLATION_MESSAGE}',
)


def build_model(prompt_enabled: bool, prompt_dictionary=PROMPT_DICTIONARY) -> EncDecRNNTModel:
    """Build a tiny RNNT model, optionally with language prompt conditioning enabled."""
    model_defaults = {'enc_hidden': ENC_HIDDEN, 'pred_hidden': 32}
    if prompt_enabled:
        model_defaults.update(
            {
                'initialize_prompt_feature': True,
                'num_prompts': NUM_PROMPTS,
                'prompt_dictionary': prompt_dictionary,
            }
        )

    cfg = DictConfig(
        {
            'labels': ListConfig([' ', 'a', 'b', 'c']),
            'preprocessor': DictConfig(
                {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': {}}
            ),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(
                {
                    'cls': 'nemo.collections.asr.modules.ConvASREncoder',
                    'params': {
                        'feat_in': 64,
                        'activation': 'relu',
                        'conv_mask': True,
                        'jasper': [
                            {
                                'filters': ENC_HIDDEN,
                                'repeat': 1,
                                'kernel': [1],
                                'stride': [1],
                                'dilation': [1],
                                'dropout': 0.0,
                                'residual': False,
                                'separable': True,
                                'se': True,
                                'se_context_size': -1,
                            }
                        ],
                    },
                }
            ),
            'decoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                    'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
                }
            ),
            'joint': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.RNNTJoint',
                    'jointnet': {'joint_hidden': 32, 'activation': 'relu'},
                }
            ),
            'decoding': DictConfig({'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}}),
            'loss': DictConfig({'loss_name': 'default'}),
        }
    )
    return EncDecRNNTModel(cfg=cfg)


@pytest.fixture()
def prompt_model():
    return build_model(prompt_enabled=True)


@pytest.fixture()
def plain_model():
    return build_model(prompt_enabled=False)


@requires_numba
class TestLangPromptSetup:
    @pytest.mark.unit
    def test_disabled_by_default(self, plain_model):
        assert plain_model.use_prompt is False
        assert plain_model.num_prompts is None
        assert plain_model.prompt_dictionary is None
        assert not hasattr(plain_model, 'prompt_kernel')
        assert 'prompt' not in plain_model.input_types

    @pytest.mark.unit
    def test_enabled_from_config(self, prompt_model):
        assert prompt_model.use_prompt is True
        assert prompt_model.num_prompts == NUM_PROMPTS
        assert prompt_model.prompt_dictionary == PROMPT_DICTIONARY
        assert prompt_model.prompt_kernel[0].in_features == NUM_PROMPTS + ENC_HIDDEN
        assert prompt_model.prompt_kernel[-1].out_features == ENC_HIDDEN
        assert 'prompt' in prompt_model.input_types

    @pytest.mark.unit
    def test_prompt_kernel_weights_are_in_state_dict(self, prompt_model):
        keys = [key for key in prompt_model.state_dict() if key.startswith('prompt_kernel.')]
        assert keys, "prompt_kernel must be a registered submodule so checkpoints round-trip"

    @pytest.mark.unit
    def test_missing_prompt_dictionary_raises(self):
        with pytest.raises(ValueError, match='prompt_dictionary'):
            build_model(prompt_enabled=True, prompt_dictionary={})


@requires_numba
class TestResolvePromptId:
    @pytest.mark.unit
    def test_known_language(self, prompt_model):
        assert prompt_model.resolve_prompt_id('de-DE') == 3
        assert prompt_model.resolve_prompt_id('en') == 0

    @pytest.mark.unit
    @pytest.mark.parametrize('target_lang', [None, 'kl-KL'])
    def test_falls_back_to_unk(self, prompt_model, target_lang):
        assert prompt_model.resolve_prompt_id(target_lang) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_missing_unk_raises(self):
        model = build_model(prompt_enabled=True, prompt_dictionary={'en-US': 0})
        with pytest.raises(ValueError, match="No 'unk' entry"):
            model.resolve_prompt_id(None)

    @pytest.mark.unit
    def test_raises_for_non_prompt_model(self, plain_model):
        with pytest.raises(ValueError, match='not trained with prompt conditioning'):
            plain_model.resolve_prompt_id('en-US')


@requires_numba
class TestApplyLangPrompt:
    @pytest.mark.unit
    def test_onehot_prompt_shape_and_content(self, prompt_model):
        prompt = prompt_model.create_onehot_prompt(3, 5, dtype=torch.float32, device=torch.device('cpu'))
        assert prompt.shape == (3, NUM_PROMPTS)
        assert torch.equal(prompt.sum(dim=-1), torch.ones(3))
        assert torch.all(prompt[:, 5] == 1.0)

    @pytest.mark.unit
    @pytest.mark.parametrize('time_steps', [1, 7, 13, 91])
    def test_broadcasts_over_any_encoder_length(self, prompt_model, time_steps):
        """A (B, num_prompts) prompt must fit any encoder length.

        Regression: callers used to precompute the time dimension from the feature length, which is
        off by one whenever that length is not a multiple of the subsampling factor. The shortfall was
        silently zero-padded, conditioning the tail of every chunk on an all-zero prompt.
        """
        encoded = torch.randn(2, ENC_HIDDEN, time_steps)
        prompt = prompt_model.create_onehot_prompt(2, 3, dtype=encoded.dtype, device=encoded.device)

        out = prompt_model.apply_lang_prompt(encoded, prompt)

        assert out.shape == (2, ENC_HIDDEN, time_steps)

    @pytest.mark.unit
    def test_per_frame_prompt_with_matching_length_is_accepted(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt = torch.zeros(2, 9, NUM_PROMPTS)
        prompt[:, :, 3] = 1.0

        out = prompt_model.apply_lang_prompt(encoded, prompt)

        assert out.shape == (2, ENC_HIDDEN, 9)
        # Broadcasting a 2D prompt must be equivalent to an explicit per-frame prompt.
        broadcast = prompt_model.apply_lang_prompt(
            encoded, prompt_model.create_onehot_prompt(2, 3, dtype=encoded.dtype, device=encoded.device)
        )
        assert torch.allclose(out, broadcast)

    @pytest.mark.unit
    @pytest.mark.parametrize('prompt_time_steps', [8, 10])
    def test_per_frame_prompt_length_mismatch_raises(self, prompt_model, prompt_time_steps):
        """Silently padding/truncating a per-frame prompt is what produced the tail-frame bug."""
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt = torch.zeros(2, prompt_time_steps, NUM_PROMPTS)
        prompt[:, :, 3] = 1.0

        with pytest.raises(ValueError, match='time steps'):
            prompt_model.apply_lang_prompt(encoded, prompt)

    @pytest.mark.unit
    def test_wrong_num_prompts_raises(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        with pytest.raises(ValueError, match='classes'):
            prompt_model.apply_lang_prompt(encoded, torch.zeros(2, NUM_PROMPTS + 1))

    @pytest.mark.unit
    def test_different_languages_give_different_output(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        out_en = prompt_model.apply_lang_prompt(
            encoded, prompt_model.create_onehot_prompt(2, 0, torch.float32, encoded.device)
        )
        out_ja = prompt_model.apply_lang_prompt(
            encoded, prompt_model.create_onehot_prompt(2, 5, torch.float32, encoded.device)
        )
        assert not torch.allclose(out_en, out_ja)

    @pytest.mark.unit
    def test_transcribe_helper_is_noop_without_prompt_support(self, plain_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        assert plain_model.apply_lang_prompt_for_transcribe(encoded, 'en-US') is encoded

    @pytest.mark.unit
    def test_transcribe_helper_conditions_on_target_lang(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        out_de = prompt_model.apply_lang_prompt_for_transcribe(encoded, 'de-DE')
        out_ja = prompt_model.apply_lang_prompt_for_transcribe(encoded, 'ja-JP')
        assert out_de.shape == encoded.shape
        assert not torch.allclose(out_de, out_ja)

    @pytest.mark.unit
    def test_prompt_rejected_by_non_prompt_model(self, plain_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        with pytest.raises(ValueError, match='not trained with prompt conditioning'):
            plain_model.apply_lang_prompt(encoded, torch.zeros(2, NUM_PROMPTS))


@requires_numba
class TestPromptPrecision:
    """The prompt projection must survive a model-wide cast to a low-precision dtype.

    The one-hot prompt is a unit-magnitude feature next to much larger encoder activations, so in
    bfloat16 its contribution falls below the mantissa and decoding silently ignores the requested
    language.
    """

    @pytest.mark.unit
    def test_kernel_stays_float32_after_model_cast(self, prompt_model):
        prompt_model.to(torch.bfloat16)

        assert all(param.dtype is torch.float32 for param in prompt_model.prompt_kernel.parameters())
        # ... while the rest of the model did get cast.
        assert prompt_model.encoder.encoder[0].mconv[0].conv.weight.dtype is torch.bfloat16

    @pytest.mark.unit
    def test_output_keeps_encoder_dtype(self, prompt_model):
        prompt_model.to(torch.bfloat16)
        encoded = torch.randn(2, ENC_HIDDEN, 9, dtype=torch.bfloat16)

        out = prompt_model.apply_lang_prompt(
            encoded, prompt_model.create_onehot_prompt(2, 3, torch.bfloat16, encoded.device)
        )

        assert out.dtype is torch.bfloat16

    @pytest.mark.unit
    def test_bfloat16_model_matches_float32_projection_exactly(self, prompt_model):
        """After casting the model to bfloat16 the projection must still compute in float32.

        Feeding bfloat16-representable activations means the only difference left between a
        float32 model and a bfloat16 one is the projection's own arithmetic plus the final cast of
        the result. So the bfloat16 model's output must equal the float32 output rounded to
        bfloat16, bit for bit. A projection that followed the model-wide cast would not match:
        the hidden layer and ReLU would be evaluated in bfloat16 too.
        """
        torch.manual_seed(0)
        # Encoder activations an order of magnitude larger than the one-hot prompt, chosen to be
        # exactly representable in bfloat16 so they are not themselves a source of difference.
        encoded = (torch.randn(2, ENC_HIDDEN, 32) * 10.0).to(torch.bfloat16)

        def project(dtype):
            prompt = prompt_model.create_onehot_prompt(2, 3, dtype, encoded.device)
            return prompt_model.apply_lang_prompt(encoded.to(dtype), prompt)

        reference = project(torch.float32)
        prompt_model.to(torch.bfloat16)
        measured = project(torch.bfloat16)

        assert measured.dtype is torch.bfloat16
        assert torch.equal(measured, reference.to(torch.bfloat16))

    @pytest.mark.unit
    def test_languages_stay_distinguishable_in_bfloat16(self, prompt_model):
        torch.manual_seed(0)
        prompt_model.to(torch.bfloat16)
        encoded = (torch.randn(2, ENC_HIDDEN, 32) * 10.0).to(torch.bfloat16)

        out_en = prompt_model.apply_lang_prompt(
            encoded, prompt_model.create_onehot_prompt(2, 0, torch.bfloat16, encoded.device)
        )
        out_ja = prompt_model.apply_lang_prompt(
            encoded, prompt_model.create_onehot_prompt(2, 5, torch.bfloat16, encoded.device)
        )

        assert not torch.equal(out_en, out_ja)

    @pytest.mark.unit
    def test_device_moves_are_still_honoured(self, prompt_model):
        prompt_model.to(torch.device('cpu'))
        assert all(param.device.type == 'cpu' for param in prompt_model.prompt_kernel.parameters())


@requires_numba
class TestForwardIntegration:
    @pytest.mark.unit
    def test_forward_accepts_prompt(self, prompt_model):
        prompt_model.eval()
        signal = torch.randn(2, 16000)
        signal_len = torch.tensor([16000, 16000])
        prompt = prompt_model.create_onehot_prompt(2, 3, torch.float32, signal.device)

        with torch.no_grad():
            encoded, encoded_len = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt=prompt
            )
            baseline, _ = prompt_model.forward(input_signal=signal, input_signal_length=signal_len)

        assert encoded.shape == baseline.shape
        assert encoded_len.shape == (2,)
        assert not torch.allclose(encoded, baseline)


class TestBufferedWrapperPromptShape:
    """``encode_with_prompts`` must send each model family the prompt shape its forward expects."""

    @staticmethod
    def _capture_prompt(use_prompt: bool):
        from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper

        captured = {}

        class Stub:
            asr_model = SimpleNamespace(use_prompt=use_prompt)

            def get_subsampling_factor(self):
                return 8

            def encode(self, processed_signal, processed_signal_length, prompt_vectors=None):
                captured['shape'] = tuple(prompt_vectors.shape)
                return processed_signal, processed_signal_length

        # 100 feature frames is deliberately not a multiple of the subsampling factor.
        RNNTInferenceWrapper.encode_with_prompts(
            Stub(), torch.zeros(2, 80, 100), torch.tensor([100, 100]), torch.zeros(2, NUM_PROMPTS)
        )
        return captured['shape']

    @pytest.mark.unit
    def test_unified_model_gets_a_time_free_prompt(self):
        """Broadcasting inside the model avoids estimating the encoder length from the feature length."""
        assert self._capture_prompt(use_prompt=True) == (2, NUM_PROMPTS)

    @pytest.mark.unit
    def test_prompt_streaming_model_still_gets_an_expanded_prompt(self):
        """`concat` models declare `prompt` as (B, T, D) and reject a 2D tensor."""
        assert self._capture_prompt(use_prompt=False) == (2, 100 // 8, NUM_PROMPTS)


class TestCacheAwareRefusal:
    @pytest.mark.unit
    def test_cache_aware_rejects_prompt_conditioned_model(self):
        """These models were not trained under pure streaming conditions.

        The cache-aware path gates prompt injection on ``concat``, which they never set, so it used
        to skip the trained prompt projection entirely and return plausible but wrong transcripts.
        """
        from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
            CacheAwareRNNTInferenceWrapper,
        )

        stub = SimpleNamespace(asr_model=SimpleNamespace(use_prompt=True))

        with pytest.raises(ValueError, match='not supported by cache-aware streaming'):
            CacheAwareRNNTInferenceWrapper._validate_prompt_support(stub)
