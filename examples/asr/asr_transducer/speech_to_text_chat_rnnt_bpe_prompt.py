# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Training a cache-aware streaming CHAT (Chunk-wise Attention Transducer) model with language-prompt
# conditioning for multilingual ASR.
#
# The model is a standard RNNT BPE model whose joint is `RNNTAttJoint` (CHAT), with an additional
# 1-hot language prompt fused into the encoder output. The target language is read from the manifest
# field given by `model.train_ds.prompt_field` (default: target_lang).

# Manifest file example (one JSON per line):
{"audio_filepath":"/data/14403911078888113072.wav","duration":12.12,"text":"the u n also hopes ...","lang":"en-US","target_lang":"en-US"}
{"audio_filepath":"/data/7340766134437057689.wav","duration":9.12,"text":"soon officers equipped ...","lang":"de-DE","target_lang":"de-DE"}


# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_chat_rnnt_bpe_prompt.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=5.0 \
    model.optim.sched.warmup_steps=10000 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

"""

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(
    config_path="../conf/fastconformer/cache_aware_streaming/",
    config_name="fastconformer_chat_transducer_bpe_streaming_prompt",
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    # EncDecRNNTBPEModel enables language-prompt conditioning automatically when the config sets
    # model.model_defaults.initialize_prompt_feature (and provides a prompt_dictionary).
    asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
