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

import json
import logging
import math
import re
from pathlib import Path
from typing import List, Mapping, Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.gpt.data.utils import (
    _chat_preprocess,
    _get_samples_mapping,
    _JSONLMemMapDataset,
    _OnlineSampleMapping,
    _preprocess,
)
from nemo.core.classes import Dataset
from nemo.lightning.base import NEMO_DATASETS_CACHE

logger = logging.getLogger(__name__)

# hack to avoid the "not enough disk space" error in some slurm cluster
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

__idx_version__ = "0.2"  # index file version
__idx_suffix__ = "idx"  # index file suffix


def get_dataset_root(name: str) -> Path:
    """ """
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    add_sep: bool = False,
    seed: int = 1234,
    label_key: str = "output",
    answer_only_loss: bool = True,
    truncation_field: str = "input",
    pad_to_max_length: bool = False,
    index_mapping_dir: Optional[str] = None,
    prompt_template: str = "{input} {output}",
    truncation_method: str = "right",
    memmap_workers: int = 2,
    hf_dataset: bool = False,
    global_sample_mapping: bool = False,
    pack_metadata_file_path: Path = None,
    pad_cu_seqlens: bool = False,
    chat: bool = False,
    use_hf_tokenizer_chat_template: bool = False,
    **kwargs,
) -> "GPTSFTDataset":
    """
    Create the dataset class (GPTSFTDataset, GPTSFTChatDataset or GPTSFTPackedDataset)
    """

    if isinstance(path, str):
        path = Path(path)

    gpt_sft_dataset_kwargs = {
        "file_path": str(path),
        "tokenizer": tokenizer,
        "max_seq_length": seq_length,
        "memmap_workers": memmap_workers,
        "hf_dataset": hf_dataset,
        "global_sample_mapping": global_sample_mapping,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "add_sep": add_sep,
        "seed": seed,
        "label_key": label_key,
        "answer_only_loss": answer_only_loss,
        "truncation_field": truncation_field,
        "pad_to_max_length": pad_to_max_length,
        "index_mapping_dir": index_mapping_dir,
        "prompt_template": prompt_template,
        "truncation_method": truncation_method,
    }

    if path.suffix == ".npy":
        return GPTSFTPackedDataset(
            pack_metadata_file_path=pack_metadata_file_path,
            pad_cu_seqlens=pad_cu_seqlens,
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )
    elif chat:
        return GPTSFTChatDataset(
            **gpt_sft_dataset_kwargs,
            use_hf_tokenizer_chat_template=use_hf_tokenizer_chat_template,
            **kwargs,
        )
    else:
        return GPTSFTDataset(
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )


class GPTSFTDataset(Dataset):
    """ """

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        global_sample_mapping: bool = False,
        truncation_method: str = "right",
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
        sanity_check_dist_workers: bool = True,
    ):
        """
        file_path: Path to a JSONL GPT supervised fine-tuning dataset.
            Data is formatted as multiple JSON lines with each line formatted as follows:
            {
                'input': 'John von Neumann\nVon Neumann made fundamental contributions ...
                    Q: What did the math of artificial viscosity do?',
                'output': 'smoothed the shock transition without sacrificing basic physics'
            }
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples.
            Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset.
            Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load.
            This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        label_key: Key to use for the label in your JSONL file
        answer_only_loss: If True, will compute the loss only on the answer part of the input.
            If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: keys in prompt_template).
            Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length.
            If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to.
            If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring.
            Formatted like Q: {context_key}\n\nA: {label_key}
        hf_dataset: Whether to load the json file with the HuggingFace dataset.
            Otherwise, will load the jsonl file with the JSONLMemMapDataset.
        global_sample_mapping: Whether to shuffle all data together, or shuffle the dataset within each epoch
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}.
            Default: {
                'system_turn_start': '<extra_id_0>',
                'turn_start': '<extra_id_1>',
                'label_start': '<extra_id_2>',
                'end_of_turn': '\n',
                'end_of_name': '\n'
            }
        is_test: Whether this dataset is the test split.
        output_original_text (bool): if true, will keep the original text in the output alongside the tokenized ids.
        sanity_check_dist_workers (bool): if true, will run sanity check across workers when making mapping.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.pad_seq_length_to_mult = pad_seq_length_to_mult
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.label_key = label_key
        self.answer_only_loss = answer_only_loss
        self.truncation_fields = truncation_field.split(",") if truncation_field is not None else []
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.memmap_workers = memmap_workers
        self.hf_dataset = hf_dataset
        self.global_sample_mapping = global_sample_mapping
        self.truncation_method = truncation_method
        self.is_test = is_test
        self.output_original_text = output_original_text
        self.ceil_to_power_2 = ceil_to_power_2
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion
        self.sanity_check_dist_workers = sanity_check_dist_workers

        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens

        self._load_dataset()

        # Validate prompt template
        self._maybe_validate_prompt_template()

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()

    def _load_dataset(self):
        if self.hf_dataset:
            self.indexed_dataset = load_dataset(
                "json",
                data_files=self.file_path,
                cache_dir=self.index_mapping_dir,
                num_proc=self.memmap_workers,
                split="train",
            )
        else:
            self.indexed_dataset = _JSONLMemMapDataset(
                dataset_paths=[self.file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=self.index_mapping_dir,
                workers=self.memmap_workers,
            )

    def _maybe_validate_prompt_template(self):
        assert (
            self.prompt_template is not None
        ), f"we need prompt_template to combine contexts and label {self.label_key}"
        # When providing things like newlines in the prompt template via the CLI, they are escaped.
        # This line unescapes them.
        self.prompt_template = self.prompt_template.encode("utf-8").decode("unicode_escape")
        self.prompt_template_keys = re.findall(r"{(.*?)}", self.prompt_template)

        label_placeholder = f"{{{self.label_key}}}"
        assert (
            self.prompt_template[-len(label_placeholder) :] == label_placeholder
        ), f"{label_placeholder} must be at the end of prompt_template."

        # Legacy checkpoints has self.truncation_fields = ['context']
        # and self.prompt_template_keys = ['input', 'output']
        if len(self.truncation_fields) > 0:
            if self.prompt_template_keys[0] == "input" and self.truncation_fields[0] == "context":
                self.truncation_fields[0] = self.prompt_template_keys[0]

        assert set(self.truncation_fields).issubset(
            self.prompt_template_keys
        ), f"truncation_fields {self.truncation_fields} must in {self.prompt_template_keys}"

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            osm = (
                _OnlineSampleMapping(
                    dataset_size=len(self.indexed_dataset),
                    num_samples=self.max_num_samples,
                )
                if not self.global_sample_mapping
                else None
            )
            self.samples_mapping = _get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split("/")[-1],
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
                samples_mapping=osm,
                sanity_check_dist_workers=self.sanity_check_dist_workers,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, (np.uint32, np.int64)):
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        # idx may < 0 because we pad_samples_to_global_batch_size, e.g. id = -1
        if idx < 0:
            idx = len(self) + idx
            auto_gen_idx = True
        else:
            auto_gen_idx = False
        try:
            example = self.indexed_dataset[idx]
            if auto_gen_idx:
                example["__AUTOGENERATED__"] = True
        except Exception as e:
            logger.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e
        return self._process_example(example)

    def _separate_template(self, prompt_template_values: List[str]):
        """
        Combine contexts and label based on prompt_template into a list of strings and a list of keys.

        Args:
            prompt_template_values (List[str]): the list of context and label strings
                extrated from jsonl file with prompt_template_keys.

        Returns:
            template_strings (List[str]): separated prompt_template with contexts/label
                placeholder filled with corresponding strings
            template_strings_keys (List[str]): strings point to placeholder keys or <template>

        Examples:
            prompt_template = 'Context:  {context} Question: {question} Answer: {label}'
            prompt_template_values = ['xxx', 'yyy', 'zzz']

            # tokenizer.space_sensitive = True
            template_strings = ['Context:', '  xxx', ' Question:', ' yyy', ' Answer:', ' zzz']

            # tokenizer.space_sensitive = False
            template_strings = ['Context:', ' xxx', 'Question:', 'yyy', 'Answer:', 'zzz']

            template_strings_keys = ['<template>', 'context', '<template>', 'question', '<template>', 'label']
        """
        placeholders = [f"{{{k}}}" for k in self.prompt_template_keys]

        # placeholder to string
        ph_to_s = {ph: s for ph, s in zip(placeholders, prompt_template_values)}
        # placeholder to key
        ph_to_k = {ph: k for ph, k in zip(placeholders, self.prompt_template_keys)}

        # separate prompt_template based on '<space>{placeholder}'
        # examples:
        #   self.prompt_template = "Context:{context}  Passage: {passage}\n\nQuestion:{question} {label}"
        #   template_with_placeholder_separated = [
        #       'Context:', '{context}', '  Passage:', ' {passage}', '\n\nQuestion:', '{question}', ' {label}'
        #   ]
        template_with_placeholder_separated = re.split("( *?{.+?})", self.prompt_template)
        template_with_placeholder_separated = [s for s in template_with_placeholder_separated if len(s) > 0]

        # remove space if we have leading space and tokenizer is not space_sensitive
        # space_sensitive = True : tokenizer.text_to_tokens('A{num_spaces}B') = (
        #   tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces}B'
        # )
        # space_sensitive = False: tokenizer.text_to_tokens('A{num_spaces}B') = (
        # tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces-1}B'
        # )
        space_sensitive = getattr(self.tokenizer, "space_sensitive", False)
        template_with_space_reduced = [
            s[1:] if not space_sensitive and s[0] == " " else s for s in template_with_placeholder_separated
        ]

        # convert placeholder to the corresponding string (preserve left spaces) and key
        template_strings, template_strings_keys = [], []
        for t in template_with_space_reduced:
            placeholder = t.lstrip(" ")
            left_spaces = " " * (len(t) - len(placeholder))
            template_strings.append(left_spaces + ph_to_s.get(placeholder, placeholder))
            template_strings_keys.append(ph_to_k.get(placeholder, "<template>"))

        return template_strings, template_strings_keys

    def _multiple_truncation(self, template_ids: List[List[int]], template_ids_keys: List[str]):
        """
        Calculate total tokens and truncate multiple contexts in truncation_fields.

        Args:
            template_ids (List[List[int]]): the list of separate prompt_template ids.
            template_ids_keys (List[str]): the list of placeholder keys or <template>
                (used to check key in truncation_fields).

        Returns:
            context_ids (List[int]): all context ids.
            label_ids (List[int]): all label ids.
        """
        context_ids = template_ids[:-1]
        label_ids = template_ids[-1]
        total_ids = (
            self.virtual_tokens
            + sum(len(ids) for ids in context_ids)
            + max(len(label_ids), self.tokens_to_generate)
            + self.add_bos
            + self.add_sep
            + self.add_eos  # Only training need to consider eos token
        )

        if total_ids > self.max_seq_length:
            truncation_length_total = total_ids - self.max_seq_length
            num_fields = len(self.truncation_fields)
            if num_fields > 0:
                # sorted equal divide length to each field
                # examples:
                #   truncation_length_total = 3
                #   num_fields = 11
                #   truncation_length_list = [3,4,4]
                truncation_length_list = [
                    truncation_length_total // num_fields + (1 if i < truncation_length_total % num_fields else 0)
                    for i in range(num_fields)[::-1]
                ]

                for i, (ids, key) in enumerate(zip(template_ids, template_ids_keys)):
                    if key in self.truncation_fields:
                        truncation_length = truncation_length_list.pop()
                        if len(ids) < truncation_length:
                            logger.warning(f"{key} is not long enough to truncate.")
                            truncation_length = len(ids)

                        truncation_length_total -= truncation_length
                        template_ids[i] = self._truncation(ids, len(ids) - truncation_length)

            if truncation_length_total > 0:
                template_ids_lengths = [len(ids) for ids in template_ids]
                if self.truncation_method == "left":
                    iters = range(0, len(template_ids_lengths), 1)
                elif self.truncation_method == "right":
                    iters = range(len(template_ids_lengths) - 1, -1, -1)
                    # We need to truncate more to let context_ids + tokens_to_generate < self.max_seq_length
                    truncation_length_total += min(len(label_ids), self.tokens_to_generate)
                else:
                    raise ValueError(f"{self.truncation_method} is not supported")

                # Iterate all lengths of template_ids.
                for i in iters:
                    if template_ids_lengths[i] >= truncation_length_total:
                        template_ids_lengths[i] -= truncation_length_total
                        template_ids[i] = self._truncation(template_ids[i], template_ids_lengths[i])
                        break
                    else:
                        truncation_length_total -= template_ids_lengths[i]
                        template_ids_lengths[i] = 0
                        template_ids[i] = self._truncation(template_ids[i], template_ids_lengths[i])

        context_ids = [i for ids in template_ids[:-1] for i in ids]
        label_ids = template_ids[-1]
        return context_ids, label_ids

    def _truncation(self, ids, expect_length):
        if expect_length == 0:
            return []
        elif self.truncation_method == "left":
            return ids[-expect_length:]
        elif self.truncation_method == "right":
            return ids[:expect_length]
        else:
            raise ValueError(f"{self.truncation_method} is not supported")

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        prompt_template_values = []
        for c in self.prompt_template_keys:
            try:
                prompt_template_values.append(example[c].strip(" "))
            except KeyError as e:
                if c == self.label_key and self.is_test:
                    # allow missing label during testing,
                    # if user only wants to do inference without calculating metrics
                    prompt_template_values.append("")
                else:
                    raise e

        template_strings, template_strings_keys = self._separate_template(prompt_template_values)
        template_ids = [self.tokenizer.text_to_ids(s) for s in template_strings]
        context_ids, answer_ids = self._multiple_truncation(template_ids, template_strings_keys)

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            context_ids = [self.tokenizer.eos_id] * self.virtual_tokens + context_ids

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]

        input_ids = context_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in self.prompt_template_keys}
        if self.output_original_text:
            for orig_text, text_key in zip(template_strings, template_strings_keys):
                metadata[text_key] = orig_text

        processed_example = {
            "input_ids": input_ids,
            "answer_start_idx": len(context_ids),
            "context_ids": context_ids,
            "context_length": len(context_ids),
            "answer_ids": answer_ids,
            "metadata": metadata,
            "token_count": len(input_ids),
        }
        processed_example["loss_mask"] = self._build_loss_mask(processed_example)

        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        if self.ceil_to_power_2:
            # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
            return 2 ** math.ceil(math.log2(n))
        else:
            return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def _build_loss_mask(self, processed_example):
        """Pad input_ids in batch to max batch length while building loss mask"""
        input_ids = processed_example["input_ids"]
        answer_start_idx = processed_example["answer_start_idx"]
        if self.answer_only_loss:
            loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
        else:
            loss_mask = [1.0] * len(input_ids)

        return loss_mask

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`."""
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch):
        input_ids = [item["input_ids"][:-1] for item in batch]
        labels = [item["input_ids"][1:] for item in batch]
        contexts = [item["context_ids"] for item in batch]
        context_lengths = torch.LongTensor([item["context_length"] for item in batch])
        answers = [item["answer_ids"] for item in batch]
        loss_mask = [self._build_loss_mask(item)[1:] for item in batch]
        metadata = [item["metadata"] for item in batch]
        token_count = [item["token_count"] for item in batch]

        max_length = max(
            max([len(x) for x in input_ids]),
            max([len(x) for x in contexts]) + self.tokens_to_generate,
        )
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(
                self.max_seq_length,
                self._ceil_to_nearest(max_length, self.pad_seq_length_to_mult),
            )
        assert max_length <= self.max_seq_length

        attention_mask = None
        if not self.get_attention_mask_from_fusion:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer.eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer.eos_id))

        processed_batch = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "contexts": contexts,
            "context_lengths": context_lengths,
            "answers": answers,
            "metadata": metadata,
            "token_count": token_count,
        }

        if not self.get_attention_mask_from_fusion:
            processed_batch["attention_mask"] = attention_mask

        return processed_batch


class GPTSFTPackedDataset(GPTSFTDataset):
    """ """

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        return_cu_seqlen: bool = True,
        pad_cu_seqlens: bool = False,
        pack_metadata_file_path: Optional[str] = None,
        **kwargs,
    ):
        """
        file_path: See `file_path` in the parent class.
        tokenizer: See `tokenizer` in the parent class.
        return_cu_seqlen: Whether to return `cu_seqlen` to pass to the model. Having `cu_seqlen` in the model input
                enables THD attention kernel, which is the correct format for training with packed sequence to prevent
                cross-sequence attention. This flag should be True unless you have a specific use case.
        """
        np.random.seed(kwargs.get("seed", 1234))
        super().__init__(file_path, tokenizer, **kwargs)
        assert self.virtual_tokens == 0, "P-Tuning with packed sequence is not supported."
        self.return_cu_seqlen = return_cu_seqlen

        self.pad_cu_seqlens = pad_cu_seqlens
        if self.pad_cu_seqlens:
            assert (
                pack_metadata_file_path is not None
            ), "a metadata json file is required when pad_cu_seqlens is enabled"
            assert (
                self.pad_to_max_length is True
            ), "'pad_to_max_length=True' is required when pad_cu_seqlens is enabled"

        self.pack_metadata = None
        if pack_metadata_file_path is not None:
            with open(pack_metadata_file_path) as f:
                self.pack_metadata = json.load(f)

    def __getitem__(self, idx):
        if self.samples_mapping is not None:
            # assert idx < len(self.samples_mapping)
            idx = self.samples_mapping[idx]

        input_ids = self.indexed_dataset[idx]["input_ids"]
        seq_boundaries = self.indexed_dataset[idx]["seq_start_id"] + [len(input_ids)]
        loss_mask = self.indexed_dataset[idx]["loss_mask"]
        if idx < 0:
            loss_mask = [0] * len(loss_mask)
        return {
            "input_ids": input_ids,
            "seq_boundaries": seq_boundaries,
            "loss_mask": loss_mask,
        }

    def _load_dataset(self):
        try:
            self.indexed_dataset = np.load(self.file_path, allow_pickle=True)
        except Exception as e:
            logger.error(
                f"Failed to load packed dataset. The dataset should be a `.npy` file. "
                f"Please check if the packed dataset was prepared correctly. The original error was:\n {e}",
            )
            exit(1)

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            # custom samples mapping logic, following the format for unpacked sft dataset
            # Note: this is epoch-level shuffling, i.e. sampling without replacement until end of epoch, then repeat.
            # Unpacked dataset shuffles by sampling with replacement indefinitely.
            dataset_len = len(self.indexed_dataset)
            max_num_epochs = np.ceil(self.max_num_samples / dataset_len)
            indices = np.arange(dataset_len)[None, :].repeat(max_num_epochs, axis=0)
            [np.random.shuffle(x) for x in indices]
            self.samples_mapping = indices.reshape(1, -1).squeeze()[: self.max_num_samples]
        else:
            self.samples_mapping = None

    def _build_loss_mask(self, processed_example):
        seq_boundaries = processed_example["seq_boundaries"]
        if self.answer_only_loss:
            return np.concatenate(
                [
                    processed_example["loss_mask"][seq_boundaries[i] : seq_boundaries[i + 1] - 1]
                    for i in range(len(seq_boundaries) - 1)
                ]
            )
        return np.concatenate(
            [
                [
                    0 if x == self.tokenizer.eos_id else 1.0
                    for x in processed_example["input_ids"][seq_boundaries[i] : seq_boundaries[i + 1] - 1]
                ]
                for i in range(len(seq_boundaries) - 1)
            ]
        )

    def _maybe_cast_to_list(self, x):
        return [item.tolist() if isinstance(item, np.ndarray) else item for item in x]

    def collate_fn(self, batch):
        input_ids = [
            np.concatenate(
                [
                    item["input_ids"][item["seq_boundaries"][i] : item["seq_boundaries"][i + 1] - 1]
                    for i in range(len(item["seq_boundaries"]) - 1)
                ]
            )
            for item in batch
        ]
        labels = [
            np.concatenate(
                [
                    item["input_ids"][item["seq_boundaries"][i] + 1 : item["seq_boundaries"][i + 1]]
                    for i in range(len(item["seq_boundaries"]) - 1)
                ]
            )
            for item in batch
        ]

        loss_mask = [self._build_loss_mask(item) for item in batch]

        token_count = [item.shape[0] for item in input_ids]

        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            # pad to the nearest multiple of 16 for FP8 training
            # for many datasets in practice, all packed sequence lengths are very close to the
            # target length (2048, 4096, 8192), so there is very minimal padding
            max_length = max(len(length) for length in input_ids)
            max_length = min(
                self.max_seq_length,
                self._ceil_to_nearest(max_length, self.pad_seq_length_to_mult),
            )
        assert max_length <= self.max_seq_length

        position_ids: List[List[int]] = []
        cu_seqlens: List[List[int]] = []
        cu_seqlens_unpadded: List[List[int]] = []
        for item in batch:
            position_ids.append([])
            cu_seqlens.append([0])
            cu_seqlens_unpadded.append([0])
            seqlens = np.array(item["seq_boundaries"][1:]) - np.array(item["seq_boundaries"][:-1])
            for length in seqlens:
                # length minus 1 because input_ids is truncated by 1 for labels
                position_ids[-1].extend(list(range(length - 1)))
                cu_seqlens[-1].append(cu_seqlens[-1][-1] + length - 1)

            # the last seq needs to be the max seq len because rope and attn kernels expect no padding
            assert cu_seqlens[-1][-1] <= max_length

            # since data is prepadded when cp_size > 1, there may be some extra padding at the end
            # of the packed sequence. In this case, we need to add the max seq len to the end.
            if cu_seqlens[-1][-1] != max_length:
                cu_seqlens[-1].append(max_length)

            for i in range(len(item["seq_boundaries"]) - 1):
                current_seq = item["input_ids"][item["seq_boundaries"][i] : item["seq_boundaries"][i + 1] - 1]

                # since the data could be prepadded with tokenizer's eos_id,
                # we can find out the index of all the eos_id
                eos_idx = np.where(np.array(current_seq) == self.tokenizer.eos_id)

                # The second eos_id index marks the length of the original unpadded sequence if the sequence is
                # prepadded for cp_size > 1. Otherwise, there is no extra padding.
                seqlen_unpadded = eos_idx[0][0] + 1 if eos_idx[0].any() else len(current_seq)
                cu_seqlens_unpadded[-1].append(cu_seqlens_unpadded[-1][-1] + seqlen_unpadded)

            # if extra paddings are added in the packed sequence, they can't be counted as
            # actual tokens for training
            if len(cu_seqlens[-1]) > len(cu_seqlens_unpadded[-1]):
                cu_seqlens_unpadded[-1].append(cu_seqlens_unpadded[-1][-1])

            if self.pad_cu_seqlens:
                # pad cu_seqlens to a constant shape with zero length sequences
                max_samples_per_bin = max(p["max_samples_per_bin"] for p in self.pack_metadata)
                # plus 2 since cu_seqlens additionally contains 0 and may append max_length
                pad_num = max_samples_per_bin - len(cu_seqlens[-1]) + 2
                cu_seqlens[-1].extend([max_length] * pad_num)

        assert len(input_ids[0]) == len(
            position_ids[0]
        ), "Dataset problem: input_ids and position_ids lengths don't match"

        input_ids = self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        labels = self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id)
        loss_mask = self._collate_item(loss_mask, max_length=max_length, pad_id=0)
        position_ids = self._collate_item(position_ids, max_length=max_length, pad_id=0)

        processed_batch = {
            "tokens": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "loss_mask": torch.LongTensor(loss_mask),
            "position_ids": torch.LongTensor(position_ids),
            "token_count": token_count,
        }

        if self.return_cu_seqlen:
            cu_seqlens = self._collate_item(
                cu_seqlens,
                max_length=max(len(length) for length in cu_seqlens) + 1,
                pad_id=-1,
            )
            cu_seqlens_unpadded = self._collate_item(
                cu_seqlens_unpadded,
                max_length=max(len(length) for length in cu_seqlens_unpadded) + 1,
                pad_id=-1,
            )
            # Pre-generate `cu_seqlens_argmin` and `max_seqlen` as CPU tensor to avoid device-to-host copies.
            cu_seqlens = torch.IntTensor(cu_seqlens)
            cu_seqlens_argmin = torch.argmin(cu_seqlens, dim=1, keepdim=True)
            seqlens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
            max_seqlen, _ = seqlens.max(dim=1, keepdim=True)
            cu_seqlens_unpadded = torch.IntTensor(cu_seqlens_unpadded)
            cu_seqlens_unpadded_argmin = torch.argmin(cu_seqlens_unpadded, dim=1, keepdim=True)

            if self.pad_cu_seqlens:
                # If padding, use the global max seqlen, so that 'pad_cu_seqlens' is the same
                # across all batches. This is maintly used compatiblity with megatron's implementation
                # of cudagraphs, which uses the same cudagraphs over all batches.
                dataset_max_seqlen = max(p['dataset_max_seqlen'] for p in self.pack_metadata)
                min_pack_seq_len = min(p['min_packed_seqlen'] for p in self.pack_metadata)
                padding_gap = max_length - min_pack_seq_len

                # Use the larger of the two values to avoid NAN issues with attention kernel
                safe_max_seqlen = max(dataset_max_seqlen, padding_gap)
                max_seqlen = torch.IntTensor([safe_max_seqlen] * len(cu_seqlens))
            else:
                seqlens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
                max_seqlen, _ = seqlens.max(dim=1, keepdim=True)
            processed_batch.update(
                {
                    "attention_mask": torch.LongTensor(
                        [1] * len(input_ids)
                    ),  # no attention mask is needed for packed seq
                    "cu_seqlens": torch.IntTensor(cu_seqlens),  # cu_seqlens_q must be in dtype torch.int32
                    "cu_seqlens_argmin": cu_seqlens_argmin,  # only required for perf
                    "max_seqlen": max_seqlen,  # only required for perf
                    "cu_seqlens_unpadded": torch.IntTensor(cu_seqlens_unpadded),
                    "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
                }
            )
        else:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            processed_batch.update(
                {
                    "attention_mask": torch.stack(attention_mask),
                }
            )

        return processed_batch


class GPTSFTChatDataset(GPTSFTDataset):
    """
    Dataset for fine-tuning a chat model.

    Accepts conversational data in ShareGPT format. If use_hf_tokenizer_chat_template is True, the dataset will
    accept both ShareGPT and HuggingFace chat template format. In the case of ShareGPT format, it will try to convert
    to HuggingFace format.

    ShareGPT format:
    {"conversations": [{"value": "...", "from": "User"}, {"value": "...", "from": "Assistant"}]}
    HuggingFace chat template format:
    {
        "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                     {"role": "assistant", "content": "..."}]
    }
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        global_sample_mapping: bool = False,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
        sanity_check_dist_workers: bool = True,
        use_hf_tokenizer_chat_template: bool = True,
        tool_schemas: Optional[str | dict] = None,
    ):
        self.use_hf_tokenizer_chat_template = use_hf_tokenizer_chat_template
        self.tool_schemas = tool_schemas
        if isinstance(self.tool_schemas, str):
            self.tool_schemas = json.loads(self.tool_schemas)

        super().__init__(
            file_path,
            tokenizer,
            max_seq_length,
            min_seq_length,
            pad_seq_length_to_mult,
            add_bos,
            add_eos,
            add_sep,
            sep_id,
            max_num_samples,
            seed,
            label_key,
            answer_only_loss,
            truncation_field,
            pad_to_max_length,
            index_mapping_dir,
            prompt_template,
            virtual_tokens,
            tokens_to_generate,
            memmap_workers,
            hf_dataset,
            global_sample_mapping,
            truncation_method,
            special_tokens,
            is_test,
            output_original_text,
            ceil_to_power_2,
            get_attention_mask_from_fusion,
            sanity_check_dist_workers,
        )

        if (
            self.use_hf_tokenizer_chat_template
            and not hasattr(self.tokenizer, "tokenizer")
            or not hasattr(self.tokenizer.tokenizer, "apply_chat_template")
        ):
            raise ValueError("Dataset does not have a tokenizer and cannot be used as a chat dataset")

    def _maybe_validate_prompt_template(self):
        """Overrides parent to make this function a no-op."""
        pass

    def _build_samples_mapping(self):
        """Build Sample Mapping and prep for _preprocess without HF template."""
        super()._build_samples_mapping()
        if self.use_hf_tokenizer_chat_template:
            return

        LABEL_START = self.special_tokens['label_start']
        END_NAME_SIGNAL = self.special_tokens['end_of_name']

        id1 = self.tokenizer.text_to_ids(PREFIX_STR)
        id2 = self.tokenizer.text_to_ids(PREFIX_STR + LABEL_START)
        self.label_start_tokens = id2[len(id1) :]

        id1 = self.tokenizer.text_to_ids(PREFIX_STR + END_NAME_SIGNAL)
        id2 = self.tokenizer.text_to_ids(PREFIX_STR)
        self.name_end_token_ids = id1[len(id2) :]

        id1 = self.tokenizer.text_to_ids(PREFIX_STR + self.special_tokens['turn_start'])
        id2 = self.tokenizer.text_to_ids(PREFIX_STR)
        self.num_turn_start_tokens = len(id1) - len(id2)

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        if not self.use_hf_tokenizer_chat_template:
            result = _preprocess(
                example,
                self.tokenizer,
                self.name_end_token_ids,
                self.label_start_tokens,
                self.special_tokens,
                self.num_turn_start_tokens,
            )
        else:
            result = _chat_preprocess(example, self.tokenizer, self.tool_schemas)

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in ['conversations', 'messages']}

        result['metadata'] = metadata
        if self.output_original_text:
            for k in ['conversations', 'messages']:
                if k in example:
                    result['metadata'][k] = example[k]

        return result

    def collate_fn(self, batch):
        """Overrides parent function to ensure loss mask gets exported correctly."""
        # Removes the last token from each input sequence to ensure the model
        # never sees the token it is supposed to predict. This enforces an
        # autoregressive training setup where the model learns to generate
        # the next token step-by-step.
        input_ids = [item["input_ids"][:-1].tolist() for item in batch]
        # Removes the first token from each input sequence to create labels
        # that align with the model's prediction target. This ensures that
        # at time step `t`, the model's output is evaluated against the token
        # that originally followed the input at `t` in the dataset.
        labels = [item["input_ids"][1:].tolist() for item in batch]
        # Context tokens remain unchanged, representing the initial portion of
        # the sequence that serves as input to the model. This allows the model
        # to condition its predictions on prior information.
        contexts = [item["context_ids"].tolist() for item in batch]
        # Extracts the assistant's response portion of the sequence, which
        # represents the part the model is trained to generate. This helps
        # distinguish between the input prompt and the expected model output.
        answers = [item["answer_ids"].tolist() for item in batch]
        # Removes the first element from the mask to align with the shifted labels,
        # ensuring that loss is only computed for valid, predictable tokens. This
        # prevents the model from incurring loss on tokens that were never meant to
        # be predicted, such as user-provided context or padding.
        loss_mask = [item["loss_mask"][1:].tolist() for item in batch]
        # Metadata remains unchanged, carrying any additional non-token-related
        # information that might be useful for evaluation, debugging, or tracking
        # purposes.
        metadata = [item['metadata'] for item in batch]
        max_length = max(max([len(x) for x in input_ids]), max([len(x) for x in contexts]) + self.tokens_to_generate)

        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]
            for i, x in enumerate(loss_mask):
                x = torch.tensor(x)
                if x.sum().item() == 0:
                    logger.warning(
                        "Due to truncation to max_seq_length, no assistant tokens are found in sample. "
                        "Setting loss_mask to all ones."
                    )
                    loss_mask[i] = [1] * self.max_seq_length
            contexts = [x[: self.max_seq_length] for x in contexts]
            answers = [x[: self.max_seq_length] for x in answers]

        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 16))

        assert max_length <= self.max_seq_length

        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        context_lengths = torch.LongTensor([len(x) for x in contexts])
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer.eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer.eos_id))

        processed_batch = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "contexts": contexts,
            "context_lengths": context_lengths,
            "answers": answers,
            "metadata": metadata,
        }

        if not self.get_attention_mask_from_fusion:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            attention_mask = torch.stack(attention_mask)
            processed_batch["attention_mask"] = attention_mask

        return processed_batch
