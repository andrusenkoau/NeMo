# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile

import pytest
import torch
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.testing.dummies import DummyManifest, dummy_cut
from lhotse.testing.random import deterministic_rng
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.parts.submodules import multitask_beam_decoding as beam_decode
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskAED
from nemo.collections.asr.parts.utils.timestamp_utils import process_aed_timestamp_outputs
from nemo.collections.common.prompts.canary import CanaryPromptFormatter, canary
from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter, canary2
from nemo.collections.common.tokenizers import CanaryTokenizer


@pytest.fixture()
def asr_model(test_data_dir):
    preprocessor = {
        'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'params': {"window_size": 0.02, "window_stride": 0.01, "features": 64},
    }

    model_defaults = {'asr_enc_hidden': 128, 'lm_enc_hidden': 64, 'lm_dec_hidden': 64}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConformerEncoder',
        'params': {
            'feat_in': 64,
            'n_layers': 1,
            'd_model': model_defaults['asr_enc_hidden'],
            'subsampling': 'dw_striding',
            'subsampling_factor': 2,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 4,
            'conv_kernel_size': 9,
        },
    }

    transf_decoder = {
        '_target_': 'nemo.collections.asr.modules.transformer.get_nemo_transformer',
        'model_name': None,
        'pretrained': False,
        'encoder': None,
        'pre_ln_final_layer_norm': True,
        'config_dict': {
            'max_sequence_length': 512,
            'num_token_types': 0,
            'hidden_size': model_defaults['lm_dec_hidden'],
            'inner_size': 4 * model_defaults['lm_dec_hidden'],
            'num_layers': 1,
            'num_attention_heads': 2,
            'pre_ln': True,
            'vocab_size': None,
        },
    }

    head = {
        '_target_': 'nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier',
        'num_layers': 1,
        'activation': 'relu',
        'log_softmax': True,
        'hidden_size': model_defaults['lm_dec_hidden'],
        'num_classes': None,
    }

    decoding = {'strategy': 'beam', 'beam': {'beam_size': 1}}

    # os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128")
    tokenizer = {
        'dir': None,
        'type': 'agg',
        'langs': {
            'spl_tokens': {
                'dir': os.path.join(test_data_dir, "asr", "tokenizers", "canary"),
                'type': 'bpe',
            },
            'en': {
                'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"),
                'type': 'wpe',
            },
            'de': {
                'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"),
                'type': 'wpe',
            },
        },
        'custom_tokenizer': {
            '_target_': 'nemo.collections.common.tokenizers.canary_tokenizer.CanaryTokenizer',
            'tokenizers': None,
        },
    }

    optim = {
        'name': 'adamw',
        'lr': 1e-4,
    }

    loss = {
        '_target_': 'nemo.collections.common.losses.smoothed_cross_entropy.SmoothedCrossEntropyLoss',
        'label_smoothing': 0.0,
    }

    modelConfig = DictConfig(
        {
            'prompt_format': 'canary',
            'prompt_defaults': [
                {"role": "user", "slots": {"source_lang": "en", "target_lang": "en", "task": "asr", "pnc": "yes"}}
            ],
            'sample_rate': 16000,
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'transf_decoder': DictConfig(transf_decoder),
            'head': DictConfig(head),
            'tokenizer': DictConfig(tokenizer),
            'decoding': DictConfig(decoding),
            'optim': DictConfig(optim),
            'loss': DictConfig(loss),
        }
    )

    model_instance = EncDecMultiTaskModel(cfg=modelConfig)
    model_instance.configure_optimizers()
    return model_instance


class TestEncDecMultiTaskModel:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecMultiTaskModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecMultiTaskModel)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self, asr_model):
        torch.manual_seed(0)
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        asr_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=321, high=500, size=[4])

        targets = torch.randint(low=0, high=100, size=[4, 10])
        targets_len = torch.randint(low=1, high=10, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                log_probs, _, _, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1],
                    input_signal_length=length[i : i + 1],
                    transcript=targets[i : i + 1],
                    transcript_length=targets_len[i : i + 1],
                )
                logprobs_instance.append(log_probs)
            logits_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _, _, _ = asr_model.forward(
                input_signal=input_signal,
                input_signal_length=length,
                transcript=targets,
                transcript_length=targets_len,
            )

        assert logits_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-5
        diff = torch.max(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-5

    @pytest.mark.unit
    def test_training_step(self, deterministic_rng, asr_model):
        cuts = CutSet(
            [
                dummy_cut(
                    0,
                    duration=1.0,
                    with_data=True,
                    supervisions=[
                        SupervisionSegment(
                            id="cut-0", recording_id="cut-0", start=0, duration=1.0, text="short", language="en"
                        )
                    ],
                ),
                dummy_cut(
                    1,
                    duration=5.0,
                    recording_duration=5.0,
                    with_data=True,
                    supervisions=[
                        SupervisionSegment(
                            id="cut-1",
                            recording_id="cut-1",
                            start=0,
                            duration=5.0,
                            text="a very long transcript",
                            language="en",
                        )
                    ],
                ),
            ]
        )
        for c in cuts:
            c.source_lang = "en"
            c.target_lang = "en"
            c.task = "asr"
            c.pnc = "no"
        dataset = PromptedAudioToTextLhotseDataset(
            tokenizer=asr_model.tokenizer, prompt=CanaryPromptFormatter(asr_model.tokenizer)
        )
        batch = dataset[cuts]

        ans = asr_model.training_step(batch, batch_nb=0)
        assert list(ans.keys()) == ["loss"]
        assert torch.is_tensor(ans["loss"])

    @pytest.mark.unit
    def test_validation_step(self, deterministic_rng, asr_model):
        cuts = CutSet(
            [
                dummy_cut(
                    0,
                    duration=1.0,
                    with_data=True,
                    supervisions=[
                        SupervisionSegment(
                            id="cut-0", recording_id="cut-0", start=0, duration=1.0, text="short", language="en"
                        )
                    ],
                ),
                dummy_cut(
                    1,
                    duration=5.0,
                    recording_duration=5.0,
                    with_data=True,
                    supervisions=[
                        SupervisionSegment(
                            id="cut-1",
                            recording_id="cut-1",
                            start=0,
                            duration=5.0,
                            text="a very long transcript",
                            language="en",
                        )
                    ],
                ),
            ]
        )
        for c in cuts:
            c.source_lang = "en"
            c.target_lang = "en"
            c.task = "asr"
            c.pnc = "no"
        dataset = PromptedAudioToTextLhotseDataset(
            tokenizer=asr_model.tokenizer, prompt=CanaryPromptFormatter(asr_model.tokenizer)
        )
        batch = dataset[cuts]

        with torch.no_grad():
            ans = asr_model.validation_pass(batch, batch_idx=0)
        print(ans)
        assert set(ans.keys()) == set(
            [
                "val_loss",
                "val_wer",
                "val_wer_num",
                "val_wer_denom",
                "val_bleu",
                "val_bleu_pred_len",
                "val_bleu_target_len",
                "val_bleu_num",
                "val_bleu_denom",
            ]
        )

    @pytest.mark.unit
    def test_save_restore_artifact(self, asr_model):
        asr_model.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'aed_bpe.nemo')
            asr_model.save_to(path)

            new_model = EncDecMultiTaskModel.restore_from(path)
            assert isinstance(new_model, type(asr_model))

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 32 + 128 + 128

    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_save_restore_artifact_change_vocab(self, asr_model, test_data_dir):
    #     asr_model.train()
    #
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
    #         asr_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type='bpe')
    #
    #         save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
    #         asr_model.train()
    #         asr_model.save_to(save_path)
    #
    #         new_model = EncDecMultiTaskModel.restore_from(save_path)
    #         assert isinstance(new_model, type(asr_model))
    #         assert isinstance(new_model.tokenizer, tokenizers.SentencePieceTokenizer)
    #         assert new_model.model_path.endswith('_tokenizer.model')
    #         assert new_model.vocab_path.endswith('_vocab.txt')
    #         assert new_model.spe_vocab_path.endswith('_tokenizer.vocab')

    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_save_restore_artifact_agg(self, asr_model, test_data_dir):
    #     tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
    #     tok_en = {"dir": tokenizer_dir, "type": "wpe"}
    #     # the below is really an english tokenizer but we pretend it is spanish
    #     tok_es = {"dir": tokenizer_dir, "type": "wpe"}
    #     tcfg = DictConfig({"type": "agg", "langs": {"en": tok_en, "es": tok_es}})
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         asr_model.change_vocabulary(new_tokenizer_dir=tcfg, new_tokenizer_type="agg")
    #
    #         save_path = os.path.join(tmpdir, "ctc_agg.nemo")
    #         asr_model.train()
    #         asr_model.save_to(save_path)
    #
    #         new_model = EncDecMultiTaskModel.restore_from(save_path)
    #         assert isinstance(new_model, type(asr_model))
    #         assert isinstance(new_model.tokenizer, tokenizers.AggregateTokenizer)
    #
    #         # should be double
    #         assert new_model.tokenizer.tokenizer.vocab_size == 254
    #         assert len(new_model.tokenizer.tokenizer.get_vocab()) == 254

    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_vocab_change(self, test_data_dir, asr_model):
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         old_tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
    #         new_tokenizer_dir = os.path.join(tmpdir, 'tokenizer')
    #
    #         os.makedirs(new_tokenizer_dir, exist_ok=True)
    #         shutil.copy2(old_tokenizer_dir, new_tokenizer_dir)
    #
    #         nw1 = asr_model.num_weights
    #         asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
    #         # No change
    #         assert nw1 == asr_model.num_weights
    #
    #         with open(os.path.join(new_tokenizer_dir, 'vocab.txt'), 'a+') as f:
    #             f.write("!\n")
    #             f.write('$\n')
    #             f.write('@\n')
    #
    #         asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
    #
    #         # rnn embedding + joint + bias
    #         pred_embedding = 3 * (asr_model.decoder.pred_hidden)
    #         joint_joint = 3 * (asr_model.joint.joint_hidden + 1)
    #         assert asr_model.num_weights == (nw1 + (pred_embedding + joint_joint))

    @pytest.mark.unit
    def test_decoding_change(self, asr_model):
        assert isinstance(asr_model.decoding.decoding, beam_decode.TransformerAEDBeamInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 5})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.TransformerAEDBeamInfer)
        assert asr_model.decoding.decoding.search_type == "default"

    @pytest.mark.unit
    def test_prompt_change(self, asr_model):
        assert asr_model.prompt_format == 'canary'
        assert isinstance(asr_model.prompt, CanaryPromptFormatter)

        # Default change prompt
        asr_model.change_prompt()
        assert asr_model.cfg.prompt_defaults is None

        prompt_defaults = asr_model.prompt.get_default_dialog_slots()
        prompt_defaults[0]['slots']['pnc'] = 'no'
        asr_model.change_prompt(prompt_defaults=prompt_defaults)

        assert asr_model.cfg.prompt_defaults[0]['slots']['pnc'] == 'no'

    @pytest.mark.unit
    def test_prompt_change_subclass(self, asr_model):
        assert asr_model.prompt_format == 'canary'
        assert isinstance(asr_model.prompt, CanaryPromptFormatter)

        class CanaryPromptFormatterSubclass(CanaryPromptFormatter):
            NAME = "canary-unit-test-stub-format"

        # Default change prompt
        asr_model.change_prompt()
        assert asr_model.cfg.prompt_defaults is None

        prompt_defaults = asr_model.prompt.get_default_dialog_slots()
        prompt_defaults[0]['slots']['pnc'] = 'no'
        asr_model.change_prompt(prompt_format='canary-unit-test-stub-format', prompt_defaults=prompt_defaults)

        assert asr_model.cfg.prompt_format == 'canary-unit-test-stub-format'
        assert asr_model.cfg.prompt_defaults[0]['slots']['pnc'] == 'no'
        assert isinstance(asr_model.prompt, CanaryPromptFormatterSubclass)

        user_prompt = asr_model.prompt.get_default_dialog_slots()[0]
        slots = user_prompt['slots']
        slots['source_lang'] = 'en'
        slots['target_lang'] = 'en'
        slots['task'] = 'asr'
        slots['pnc'] = 'no'
        ans = asr_model.prompt.encode_dialog([user_prompt])
        recovered = asr_model.tokenizer.ids_to_text(ans["input_ids"])
        assert recovered == "<|startoftranscript|><|en|><|transcribe|><|en|><|nopnc|>"

    @pytest.mark.unit
    def test_transcribe_single_file(self, asr_model, test_data_dir):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = asr_model.transcribe(audio_file, batch_size=1)
        assert len(outputs) == 1
        assert isinstance(outputs[0].text, str)

    @pytest.mark.unit
    def test_transcribe_single_file_translation(self, asr_model, test_data_dir):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = asr_model.transcribe(audio_file, batch_size=1, task="ast", source_lang='en', target_lang='de')
        assert len(outputs) == 1
        assert isinstance(outputs[0].text, str)

    @pytest.mark.unit
    def test_transcribe_return_hypothesis(self, asr_model, test_data_dir):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = asr_model.transcribe(audio_file, batch_size=1, return_hypotheses=True)
        assert len(outputs) == 1
        assert isinstance(outputs[0], Hypothesis)

        hyp = outputs[0]
        assert isinstance(hyp.text, str)
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert hyp.alignments is None

    @pytest.mark.unit
    def test_transcribe_tensor(self, asr_model, test_data_dir):
        # Load audio file
        import soundfile as sf

        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
        audio, sr = sf.read(audio_file, dtype='float32')

        # Numpy array test
        outputs = asr_model.transcribe(audio, batch_size=1)
        assert len(outputs) == 1
        assert isinstance(outputs[0].text, str)

    @pytest.mark.unit
    def test_build_tokenizer(self, asr_model, test_data_dir):
        # Load audio file
        task_tokens = ["ast", "asr"]
        lang_tokens = ["en", "es", "de", "fr"]
        tokens = task_tokens + lang_tokens
        spl_tokenizer_from_build = CanaryTokenizer.build_special_tokenizer(tokens, test_data_dir)

        tokenizer_cfg = {'dir': os.path.join(test_data_dir), 'type': 'bpe'}
        spl_tokenizer_from_load = asr_model._make_tokenizer(tokenizer_cfg, "spl_tokens")[0]

        tokens += ["<|nospeech|>", "<pad>", "<|endoftext|>", "<|startoftranscript|>", "<|pnc|>", "<|nopnc|>"]

        ids1 = [spl_tokenizer_from_build.tokens_to_ids(t)[0] for t in tokens]
        ids2 = [spl_tokenizer_from_load.tokens_to_ids(t)[0] for t in tokens]

        for i, j in zip(ids1, ids2):
            assert i == j

    @pytest.mark.unit
    def test_predict_step(self, asr_model, test_data_dir):
        cuts = DummyManifest(CutSet, begin_id=0, end_id=1, with_data=True)
        c = cuts[0]
        c.supervisions[0].language = "en"
        c.source_lang = "en"
        c.target_lang = "en"
        c.task = "asr"
        c.pnc = "no"
        dataset = PromptedAudioToTextLhotseDataset(
            tokenizer=asr_model.tokenizer, prompt=CanaryPromptFormatter(asr_model.tokenizer)
        )
        batch = dataset[cuts]

        # Numpy array test
        outputs = asr_model.predict_step(batch)
        print(outputs)
        assert len(outputs) == 1
        assert len(outputs[0]) == 2
        assert isinstance(outputs[0][0], MonoCut)
        assert isinstance(outputs[0][1].text, str)

    @pytest.mark.unit
    def test_FrameBatchMultiTaskAED(self, asr_model, test_data_dir):
        model = FrameBatchMultiTaskAED(asr_model, batch_size=1)

        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
        meta = {
            'audio_filepath': audio_file,
            'duration': 100000,
            'source_lang': 'en',
            'taskname': 'asr',
            'target_lang': 'en',
            'pnc': 'yes',
            'answer': 'nothing',
        }
        model.read_audio_file(audio_file, delay=0.0, model_stride_in_secs=40.0, meta_data=meta)
        outputs = model.transcribe()
        assert isinstance(outputs, Hypothesis)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_FrameBatchMultiTaskAED_with_timestamps(self, canary_1b_flash):
        canary_1b_flash.eval()
        model = FrameBatchMultiTaskAED(
            canary_1b_flash,
            frame_len=10.0,
            total_buffer=10.0,
            batch_size=8,
        )

        audio_file = "/home/TestData/asr/longform/earnings22/sample_4469669.wav"
        meta = {
            'audio_filepath': audio_file,
            'duration': 100000,
            'source_lang': 'en',
            'taskname': 'asr',
            'target_lang': 'en',
            'pnc': 'yes',
            'answer': 'nothing',
            'timestamp': 'yes',
        }
        model_stride_in_secs = 0.01 * 8  # feature_stride in sec * model_stride
        model.read_audio_file(audio_file, delay=0.0, model_stride_in_secs=model_stride_in_secs, meta_data=meta)
        outputs = model.transcribe()

        # check hypothesis object
        assert isinstance(outputs, Hypothesis)

        # check part of transcript
        assert outputs.text[:13] == "Now it's time", f"{outputs}"

        # check timestamps
        assert outputs.timestamp['segment'][0]['start'] == pytest.approx(5.68)
        assert outputs.timestamp['segment'][0]['end'] == pytest.approx(9.68)


@pytest.mark.unit
def test_prompted_dataset(asr_model):
    dataset = PromptedAudioToTextLhotseDataset(
        tokenizer=asr_model.tokenizer, prompt=CanaryPromptFormatter(asr_model.tokenizer)
    )

    cuts = DummyManifest(CutSet, begin_id=0, end_id=3, with_data=True)

    c = cuts[0]
    c.supervisions[0].language = "en"
    c.source_lang = "en"
    c.target_lang = "en"
    c.task = "asr"
    c.pnc = "no"

    c = cuts[1]
    c.supervisions[0].language = "de"
    c.supervisions[0].text = "unerheblich"
    c.source_lang = "en"
    c.target_lang = "de"
    c.taskname = "ast"  # note: testing for "taskname" as we support it together with "task"
    c.pnc = "yes"

    c = cuts[2]
    c.supervisions[0].language = "en"
    c.supervisions[0].text = ""
    c.source_lang = "en"
    c.target_lang = "en"
    c.task = "asr"
    c.pnc = "yes"

    batch = dataset[cuts]

    assert isinstance(batch, PromptedAudioToTextMiniBatch)
    assert batch.audio.shape == (3, 16000)
    assert batch.audio_lens.tolist() == [16000, 16000, 16000]

    # Test example 0 (transcription)
    i = 0
    assert (
        asr_model.tokenizer.ids_to_text(batch.prompt[i]) == '<|startoftranscript|><|en|><|transcribe|><|en|><|nopnc|>'
    )
    assert batch.prompt_lens[i] == 5
    assert asr_model.tokenizer.ids_to_text(batch.transcript[i]) == 'i##r##r##el##e##v##a##nt<pad><pad>'
    assert batch.transcript_lens[i] == 8
    assert (
        asr_model.tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startoftranscript|><|en|><|transcribe|><|en|><|nopnc|>i##r##r##el##e##v##a##nt<|endoftext|><pad><pad>'
    )
    assert batch.prompted_transcript_lens[i] == 14

    # Test example 1 (translation)
    i = 1
    assert asr_model.tokenizer.ids_to_text(batch.prompt[i]) == '<|startoftranscript|><|en|><|translate|><|de|><|pnc|>'
    assert batch.prompt_lens[i] == 5
    assert asr_model.tokenizer.ids_to_text(batch.transcript[i]) == 'u##ne##r##h##e##b##l##i##c##h'
    assert batch.transcript_lens[i] == 10
    assert (
        asr_model.tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startoftranscript|><|en|><|translate|><|de|><|pnc|>u##ne##r##h##e##b##l##i##c##h<|endoftext|>'
    )
    assert batch.prompted_transcript_lens[i] == 16

    # Test example 2 (no transcript, e.g. noise)
    i = 2
    assert asr_model.tokenizer.ids_to_text(batch.prompt[i]) == '<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|>'
    assert batch.prompt_lens[i] == 5
    assert asr_model.tokenizer.ids_to_text(batch.transcript[i]) == '<pad>' * 10
    assert batch.transcript_lens[i] == 0
    assert (
        asr_model.tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|><|endoftext|>' + '<pad>' * 10
    )
    assert batch.prompted_transcript_lens[i] == 6


@pytest.fixture()
def canary2_tokenizer(asr_model, tmp_path):
    return CanaryTokenizer(
        {
            "spl_tokens": CanaryTokenizer.build_special_tokenizer(
                [
                    "<|startofcontext|>",
                    "<|en|>",
                    "<|de|>",
                    "<|pnc|>",
                    "<|nopnc|>",
                    "<|itn|>",
                    "<|noitn|>",
                    "<|diarize|>",
                    "<|nodiarize|>",
                    "<|timestamp|>",
                    "<|notimestamp|>",
                    "<|emo:undefined|>",
                    "<|emo:happy|>",
                ]
                # Timestamp frame special tokens
                + [f"<|{i}|>" for i in range(900)],
                tmp_path,
                force_rebuild=False,
            ),
            "en": asr_model.tokenizer.tokenizers_dict["en"],
            "de": asr_model.tokenizer.tokenizers_dict["de"],
        }
    )


@pytest.mark.unit
def test_prompted_dataset_canary2(canary2_tokenizer):
    dataset = PromptedAudioToTextLhotseDataset(
        tokenizer=canary2_tokenizer, prompt=Canary2PromptFormatter(canary2_tokenizer)
    )

    cuts = DummyManifest(CutSet, begin_id=0, end_id=4, with_data=True)

    # backward compatibility
    c = cuts[0]
    c.supervisions[0].language = "en"
    c.source_lang = "en"
    c.target_lang = "en"

    # new format
    c = cuts[1]
    c.supervisions[0].language = "en"
    c.supervisions[0].text = "asd"
    c.source_lang = "en"
    c.target_lang = "en"
    c.pnc = "yes"
    c.itn = "yes"
    c.diarize = "yes"
    c.timestamp = "yes"
    c.emotion = "<|emo:happy|>"
    c.decodercontext = ""

    # new format with extra context
    c = cuts[2]
    c.supervisions[0].language = "en"
    c.supervisions[0].text = "asd"
    c.source_lang = "en"
    c.target_lang = "en"
    c.pnc = "<|pnc|>"
    c.itn = "<|noitn|>"
    c.diarize = "<|diarize|>"
    c.timestamp = "<|timestamp|>"
    c.emotion = "<|emo:happy|>"
    c.decodercontext = "some decoder context"

    # transcript with timestamps
    c = cuts[3]
    c.supervisions[0].language = "en"
    c.supervisions[0].text = "<|0|> hello <|3|> <|4|> world <|5|>"
    c.source_lang = "en"
    c.target_lang = "en"
    c.pnc = "<|pnc|>"
    c.itn = "<|noitn|>"
    c.diarize = "<|diarize|>"
    c.timestamp = "<|timestamp|>"
    c.emotion = "<|emo:happy|>"
    c.decodercontext = "some decoder context"

    batch = dataset[cuts]

    assert isinstance(batch, PromptedAudioToTextMiniBatch)
    assert batch.audio.shape == (4, 16000)
    assert batch.audio_lens.tolist() == [16000, 16000, 16000, 16000]

    # Test example 0
    i = 0
    assert (
        canary2_tokenizer.ids_to_text(batch.prompt[i])
        == '<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.prompt_lens[i] == 9
    assert canary2_tokenizer.ids_to_text(batch.transcript[i]) == 'i##r##r##el##e##v##a##nt<pad><pad><pad><pad><pad>'
    assert batch.transcript_lens[i] == 8
    assert (
        canary2_tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>i##r##r##el##e##v##a##nt<|endoftext|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.prompted_transcript_lens[i] == 18

    # Test example 1
    i = 1
    assert (
        canary2_tokenizer.ids_to_text(batch.prompt[i])
        == '<|startofcontext|><|startoftranscript|><|emo:happy|><|en|><|en|><|pnc|><|itn|><|timestamp|><|diarize|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.prompt_lens[i] == 9
    assert (
        canary2_tokenizer.ids_to_text(batch.transcript[i])
        == 'a##s##d<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.transcript_lens[i] == 3
    assert (
        canary2_tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startofcontext|><|startoftranscript|><|emo:happy|><|en|><|en|><|pnc|><|itn|><|timestamp|><|diarize|>a##s##d<|endoftext|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.prompted_transcript_lens[i] == 13

    # Test example 2
    i = 2
    assert (
        canary2_tokenizer.ids_to_text(batch.prompt[i])
        == '<|startofcontext|>s##o##m##ed##e##c##o##d##erc##o##nt##e##x##t<|startoftranscript|><|emo:happy|><|en|><|en|><|pnc|><|noitn|><|timestamp|><|diarize|>'
    )
    assert batch.prompt_lens[i] == 25
    assert (
        canary2_tokenizer.ids_to_text(batch.transcript[i])
        == 'a##s##d<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.transcript_lens[i] == 3
    assert (
        canary2_tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startofcontext|>s##o##m##ed##e##c##o##d##erc##o##nt##e##x##t<|startoftranscript|><|emo:happy|><|en|><|en|><|pnc|><|noitn|><|timestamp|><|diarize|>a##s##d<|endoftext|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
    )
    assert batch.prompted_transcript_lens[i] == 29

    # Test example 3
    i = 3
    assert (
        canary2_tokenizer.ids_to_text(batch.prompt[i])
        == '<|startofcontext|>s##o##m##ed##e##c##o##d##erc##o##nt##e##x##t<|startoftranscript|><|emo:happy|><|en|><|en|><|pnc|><|noitn|><|timestamp|><|diarize|>'
    )
    assert batch.prompt_lens[i] == 25
    assert canary2_tokenizer.ids_to_text(batch.transcript[i]) == '<|0|>h##el##l##o<|3|><|4|>w##o##r##l##d<|5|>'
    assert batch.transcript_lens[i] == 13
    assert (
        canary2_tokenizer.ids_to_text(batch.prompted_transcript[i])
        == '<|startofcontext|>s##o##m##ed##e##c##o##d##erc##o##nt##e##x##t<|startoftranscript|><|emo:happy|><|en|><|en|><|pnc|><|noitn|><|timestamp|><|diarize|><|0|>h##el##l##o<|3|><|4|>w##o##r##l##d<|5|><|endoftext|>'
    )
    assert batch.prompted_transcript_lens[i] == 39


@pytest.mark.unit
def test_aed_timestamp_processing():
    # Create test hypothesis with timestamps
    hyp = Hypothesis(
        text="<|10|>hello<|15|> <|20|>world<|25|>",
        y_sequence=None,
        score=None,
        alignments=None,
        length=None,
        timestamp={},
    )

    # Process timestamps with default parameters
    processed = process_aed_timestamp_outputs(hyp)
    assert isinstance(processed, list)
    assert len(processed) == 1
    assert processed[0].text == "hello world"

    # Check word-level timestamps
    word_timestamps = processed[0].timestamp['word']
    assert len(word_timestamps) == 2

    # Check first word "hello"
    assert word_timestamps[0]['word'] == 'hello'
    assert word_timestamps[0]['start_offset'] == 10
    assert word_timestamps[0]['end_offset'] == 15
    assert word_timestamps[0]['start'] == 0.1  # 10 * 0.01
    assert word_timestamps[0]['end'] == 0.15  # 15 * 0.01

    # Check second word "world"
    assert word_timestamps[1]['word'] == 'world'
    assert word_timestamps[1]['start_offset'] == 20
    assert word_timestamps[1]['end_offset'] == 25
    assert word_timestamps[1]['start'] == 0.2  # 20 * 0.01
    assert word_timestamps[1]['end'] == 0.25  # 25 * 0.01

    # Check segment-level timestamps
    segments = processed[0].timestamp['segment']
    assert len(segments) == 1
    assert segments[0]['start_offset'] == 10
    assert segments[0]['end_offset'] == 25
    assert segments[0]['start'] == 0.1
    assert segments[0]['end'] == 0.25

    # Test with different window_stride and subsampling_factor
    hyp = Hypothesis(
        text="<|10|>hello<|15|> <|20|>world<|25|>",
        y_sequence=None,
        score=None,
        alignments=None,
        length=None,
        timestamp={},
    )
    processed = process_aed_timestamp_outputs(hyp, subsampling_factor=2, window_stride=0.02)
    word_timestamps = processed[0].timestamp['word']

    # Check timing calculations with new parameters
    assert word_timestamps[0]['start'] == 0.4  # 10 * 0.02 * 2
    assert word_timestamps[0]['end'] == 0.6  # 15 * 0.02 * 2
    assert word_timestamps[1]['start'] == 0.8  # 20 * 0.02 * 2
    assert word_timestamps[1]['end'] == 1.0  # 25 * 0.02 * 2

    # Test case when text doesn't contain timestamps
    hyp = Hypothesis(text="hello world", y_sequence=None, score=None, alignments=None, length=None, timestamp={})

    # Process timestamps with default parameters
    processed = process_aed_timestamp_outputs(hyp)
    assert isinstance(processed, list)
    assert len(processed) == 1
    assert processed[0].text == "hello world"

    # Verify no timestamps were extracted
    assert processed[0].timestamp['word'] == []
    assert processed[0].timestamp['segment'] == []
