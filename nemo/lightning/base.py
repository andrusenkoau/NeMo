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

import gc
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed
from lightning.pytorch import Trainer
from torch import nn


DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
NEMO_CACHE_HOME = Path(os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME))
DEFAULT_NEMO_DATASETS_CACHE = NEMO_CACHE_HOME / "datasets"
NEMO_DATASETS_CACHE = Path(os.getenv("NEMO_DATASETS_CACHE", DEFAULT_NEMO_DATASETS_CACHE))
DEFAULT_NEMO_MODELS_CACHE = NEMO_CACHE_HOME / "models"
NEMO_MODELS_CACHE = Path(os.getenv("NEMO_MODELS_CACHE", DEFAULT_NEMO_MODELS_CACHE))

if os.getenv('TOKENIZERS_PARALLELISM') is None:
    os.putenv('TOKENIZERS_PARALLELISM', 'True')


def get_vocab_size(
    config,
    vocab_size: int,
    make_vocab_size_divisible_by: int = 128,
) -> int:
    """returns `vocab size + padding` to make sure sum is dividable by `make_vocab_size_divisible_by`"""
    from nemo.utils import logging

    after = vocab_size
    multiple = make_vocab_size_divisible_by * config.tensor_model_parallel_size
    after = ((after + multiple - 1) // multiple) * multiple
    logging.info(
        f"Padded vocab_size: {after}, original vocab_size: {vocab_size}, dummy tokens:" f" {after - vocab_size}."
    )

    return after


def teardown(trainer: Trainer, model: Optional[nn.Module] = None) -> None:
    """Destroys distributed environment and cleans up cache / collects garbage"""
    # Destroy torch distributed
    if torch.distributed.is_initialized():
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    trainer._teardown()  # noqa: SLF001
    if model is not None:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except:
                pass

    gc.collect()
    torch.cuda.empty_cache()


__all__ = ["get_vocab_size", "teardown"]
