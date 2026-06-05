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

import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.data.audio_to_text_lhotse_prompt import LhotseSpeechToTextBpeDatasetWithPrompt
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, TranscribeConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils


@dataclass
class RNNTBPEPromptTranscribeConfig(TranscribeConfig):
    """Transcription config for RNNT BPE models with optional language-prompt conditioning."""

    target_lang: str = "en-US"
    prompt_field: str = "target_lang"


class EncDecRNNTBPEModel(EncDecRNNTModel, ASRBPEMixin):
    """Base class for encoder decoder RNNT-based models with subword tokenization.

    Optionally supports language-prompt conditioning: when
    ``model.model_defaults.initialize_prompt_feature`` is set, a 1-hot language vector is
    concatenated to the encoder output and projected back to the encoder hidden size by a
    small MLP (``prompt_kernel``) before the joint network. This is compatible with the CHAT
    (``RNNTAttJoint``) joint, whose chunking happens after the prompt is fused. When the prompt
    feature is disabled (the default), the model behaves exactly like a standard RNNT BPE model.
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_256",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_256",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_256/versions/1.6.0/files/stt_en_contextnet_256.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_512/versions/1.6.0/files/stt_en_contextnet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_1024/versions/1.9.0/files/stt_en_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_256_mls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_256_mls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_256_mls/versions/1.0.0/files/stt_en_contextnet_256_mls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_512_mls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_512_mls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_512_mls/versions/1.0.0/files/stt_en_contextnet_512_mls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_contextnet_1024_mls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_contextnet_1024_mls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_contextnet_1024_mls/versions/1.0.0/files/stt_en_contextnet_1024_mls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_small/versions/1.6.0/files/stt_en_conformer_transducer_small.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_medium/versions/1.6.0/files/stt_en_conformer_transducer_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_large/versions/1.10.0/files/stt_en_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_large_ls/versions/1.8.0/files/stt_en_conformer_transducer_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_xlarge/versions/1.10.0/files/stt_en_conformer_transducer_xlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_transducer_xxlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_transducer_xxlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_xxlarge/versions/1.8.0/files/stt_en_conformer_transducer_xxlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_contextnet_1024/versions/1.4.0/files/stt_de_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_contextnet_1024/versions/1.5/files/stt_fr_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_contextnet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_contextnet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_contextnet_1024/versions/1.8.0/files/stt_es_contextnet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_conformer_transducer_large/versions/1.5.0/files/stt_de_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_conformer_transducer_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_conformer_transducer_large/versions/1.5/files/stt_fr_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_conformer_transducer_large/versions/1.8.0/files/stt_es_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_conformer_transducer_large/versions/1.0.0/files/stt_enes_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_contextnet_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_contextnet_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_contextnet_large/versions/1.0.0/files/stt_enes_contextnet_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ca_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ca_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_conformer_transducer_large/versions/1.11.0/files/stt_ca_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_rw_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_rw_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_rw_conformer_transducer_large/versions/1.11.0/files/stt_rw_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_enes_conformer_transducer_large_codesw",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_enes_conformer_transducer_large_codesw",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_conformer_transducer_large_codesw/versions/1.0.0/files/stt_enes_conformer_transducer_large_codesw.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_kab_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_kab_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_kab_conformer_transducer_large/versions/1.12.0/files/stt_kab_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_be_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_be_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_be_conformer_transducer_large/versions/1.12.0/files/stt_be_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hr_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hr_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hr_conformer_transducer_large/versions/1.11.0/files/stt_hr_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_conformer_transducer_large/versions/1.13.0/files/stt_it_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_conformer_transducer_large/versions/1.13.0/files/stt_ru_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_eo_conformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_eo_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_eo_conformer_transducer_large/versions/1.14.0/files/stt_eo_conformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large/versions/1.0.0/files/stt_en_fastconformer_transducer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large_ls/versions/1.0.0/files/stt_en_fastconformer_transducer_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_xlarge/versions/1.20.1/files/stt_en_fastconformer_transducer_xlarge.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_transducer_xxlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_transducer_xxlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_xxlarge/versions/1.20.1/files/stt_en_fastconformer_transducer_xxlarge.nemo",
        )
        results.append(model)

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Tokenizer is necessary for this model
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        super().__init__(cfg=cfg, trainer=trainer)

        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Optional language-prompt conditioning. Disabled by default so the model behaves
        # exactly like a standard RNNT BPE model unless explicitly enabled in the config.
        self.concat = False
        model_defaults = self.cfg.get('model_defaults', None)
        self.num_prompts = model_defaults.get('num_prompts', 128) if model_defaults else 128
        if model_defaults is not None and model_defaults.get('initialize_prompt_feature', False):
            self.initialize_prompt_feature()

    def initialize_prompt_feature(self):
        """Initialize the projection that fuses a 1-hot language prompt into the encoder output."""
        if 'prompt_dictionary' not in self.cfg.model_defaults:
            raise ValueError("No prompt_dictionary found in config under model.model_defaults.")

        logging.info("RNNT BPE model: language-prompt conditioning enabled.")

        # Enable concatenation mode
        self.concat = True
        self.num_prompts = self.cfg.model_defaults.get('num_prompts', 128)

        # Projection: [encoder_hidden + num_prompts] -> [encoder_hidden]
        proj_in_size = self.num_prompts + self.cfg.model_defaults.enc_hidden
        proj_out_size = self.cfg.model_defaults.enc_hidden

        self.prompt_kernel = torch.nn.Sequential(
            torch.nn.Linear(proj_in_size, proj_out_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_out_size * 2, proj_out_size),
        )

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning
        on from pre-trained model. This method changes only decoder and leaves encoder and pre-processing
        modules unchanged. For example, you would use it if you want to use pretrained encoder when fine-tuning
        on data in another language, or when you'd need model to learn capitalization, punctuation
        and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer
                (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.

        Returns: None

        """
        if isinstance(new_tokenizer_dir, DictConfig):
            if new_tokenizer_type == 'agg':
                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer \
                        type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`, got {new_tokenizer_type}')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        joint_config = self.joint.to_config_dict()
        new_joint_config = copy.deepcopy(joint_config)
        if self.tokenizer_type == "agg":
            new_joint_config["vocabulary"] = ListConfig(vocabulary)
        else:
            new_joint_config["vocabulary"] = ListConfig(list(vocabulary.keys()))

        new_joint_config['num_classes'] = len(vocabulary)
        del self.joint
        self.joint = EncDecRNNTBPEModel.from_config_dict(new_joint_config)

        decoder_config = self.decoder.to_config_dict()
        new_decoder_config = copy.deepcopy(decoder_config)
        new_decoder_config.vocab_size = len(vocabulary)
        del self.decoder
        self.decoder = EncDecRNNTBPEModel.from_config_dict(new_decoder_config)

        del self.loss
        self.loss = RNNTLoss(num_classes=self.joint.num_classes_with_blank - 1)

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Update config
        with open_dict(self.cfg.joint):
            self.cfg.joint = new_joint_config

        with open_dict(self.cfg.decoder):
            self.cfg.decoder = new_decoder_config

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoder to output to {self.joint.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose: bool = True):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            verbose: A flag to enable/disable logging.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.joint.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            # Use the prompt-aware dataset when the dataset config provides a prompt dictionary.
            # This produces a 1-hot language prompt per encoder frame alongside the usual fields.
            if config.get("prompt_dictionary") is not None:
                logging.info("Setting up Lhotse dataset with language-prompt support")
                return get_lhotse_dataloader_from_config(
                    config,
                    global_rank=(
                        self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank")
                    ),
                    world_size=(
                        self.world_size if not config.get("do_transcribe", False) else config.get("world_size")
                    ),
                    dataset=LhotseSpeechToTextBpeDatasetWithPrompt(tokenizer=self.tokenizer, cfg=config),
                    tokenizer=self.tokenizer,
                )
            return get_lhotse_dataloader_from_config(
                config,
                # During transcription, the model is initially loaded on the CPU.
                # To ensure the correct global_rank and world_size are set,
                # these values must be passed from the configuration.
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=self.tokenizer,
                    return_cuts=config.get("do_transcribe", False),
                ),
                tokenizer=self.tokenizer,
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'use_lhotse': config.get('use_lhotse', True),
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    # ---------------------------------------------------------------------------------------------
    # Language-prompt conditioning. All methods below fall back to the standard RNNT BPE behavior
    # (via ``super()``) when prompt conditioning is disabled (``self.concat`` is False).
    # ---------------------------------------------------------------------------------------------

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()

        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "prompt": NeuralType(('B', 'T', 'D'), LabelsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        prompt=None,
    ):
        """Acoustic (encoder) forward, optionally conditioned on a language prompt.

        When prompt conditioning is enabled and a prompt is provided, the 1-hot language prompt is
        concatenated to the encoder output and projected back to the encoder hidden size before being
        returned, so the (CHAT) joint network sees prompt-conditioned features. Otherwise this is
        identical to the standard RNNT encoder forward.

        Args:
            input_signal: Raw audio of shape [B, T].
            input_signal_length: Lengths of the audio sequences, shape [B].
            processed_signal: Pre-processed features of shape [B, D, T].
            processed_signal_length: Lengths of the processed sequences, shape [B].
            prompt: 1-hot language prompt of shape [B, T, num_prompts] (only used when enabled).

        Returns:
            A tuple (encoded [B, D, T], encoded_len [B]).
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        if self.concat and prompt is not None:
            encoded = torch.transpose(encoded, 1, 2)  # B * D * T -> B * T * D
            if prompt.shape[1] > encoded.shape[1]:
                prompt = prompt[:, : encoded.shape[1], :]
            out_dtype = encoded.dtype

            # Concatenate encoder states with the language prompt and project back to enc_hidden.
            concat_enc_states = torch.cat([encoded, prompt], dim=-1)
            encoded = self.prompt_kernel(concat_enc_states).to(out_dtype)
            encoded = torch.transpose(encoded, 1, 2)  # B * T * D -> B * D * T

        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        if not self.concat:
            return super().training_step(batch, batch_nb)

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, prompt = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, prompt=prompt)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, encoder_lengths=encoded_len)
            # For CHAT, the joint chunks the encoder output; the loss must use the chunk count.
            effective_len = getattr(self.joint, 'num_chunks_per_utterance', encoded_len)
            loss_value = self.loss(
                log_probs=joint,
                targets=transcript,
                input_lengths=effective_len,
                target_lengths=target_length,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if not self.concat:
            return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

        signal, signal_len, transcript, transcript_len, prompt = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, prompt=prompt)
        del signal

        best_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        batch_size = signal_len.shape[0]
        sample_id = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size).cpu().detach().numpy()

        return list(zip(sample_id, best_hyp))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if not self.concat:
            return super().validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)

        signal, signal_len, transcript, transcript_len, prompt = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, prompt=prompt)
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, encoder_lengths=encoded_len)
                effective_len = getattr(self.joint, 'num_chunks_per_utterance', encoded_len)
                loss_value = self.loss(
                    log_probs=joint,
                    targets=transcript,
                    input_lengths=effective_len,
                    target_lengths=target_length,
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig) -> dict:
        """Encoder forward for transcription, building (or reusing) the language prompt when enabled."""
        if not self.concat:
            return super()._transcribe_forward(batch, trcfg)

        audio, audio_lens = batch[0], batch[1]

        # Reuse a prompt supplied by the dataloader if present, otherwise build it dynamically.
        prompt = batch[4] if len(batch) >= 5 else None

        if prompt is None:
            target_lang = getattr(trcfg, 'target_lang', 'en-US')
            prompt_dict = self.cfg.model_defaults.get('prompt_dictionary')
            num_prompts = self.cfg.model_defaults.get('num_prompts', 128)

            if not prompt_dict:
                raise ValueError("Prompt dictionary is empty. Cannot create dynamic prompts.")
            if target_lang not in prompt_dict:
                available_keys = list(prompt_dict.keys())
                raise ValueError(
                    f"Unknown target language: '{target_lang}'. Available languages: "
                    f"{available_keys[:10]}{'...' if len(available_keys) > 10 else ''}"
                )

            prompt_id = prompt_dict[target_lang]

            processed_signal, processed_signal_length = self.preprocessor(input_signal=audio, length=audio_lens)
            time_length = processed_signal.shape[2]
            subsampling_factor = self.cfg.encoder.get('subsampling_factor', 8)
            hidden_length = math.ceil(time_length / subsampling_factor)

            prompt = torch.zeros(audio.shape[0], hidden_length, num_prompts, dtype=torch.float32, device=audio.device)
            prompt[:, :, prompt_id] = 1.0

            encoded, encoded_len = self.forward(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length, prompt=prompt
            )
        else:
            encoded, encoded_len = self.forward(input_signal=audio, input_signal_length=audio_lens, prompt=prompt)

        return dict(encoded=encoded, encoded_len=encoded_len)
