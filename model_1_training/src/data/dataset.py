"""
dataset.py

PyTorch Dataset that converts Format A integer sequences into
tokenized inputs for RoBERTa sequence classification.

Each row's `sequence_ids` (list of infrastructure token ints) is mapped to
special token strings via the vocabulary, then encoded by the tokenizer.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from model_1_training.src.data.tokenizer_setup import sequence_ids_to_string


class TrackADataset(Dataset):
    """
    PyTorch Dataset for Track A anomaly classification.

    Converts raw integer sequence IDs into tokenizer-encoded tensors
    suitable for RoBERTa.
    """

    def __init__(
        self,
        sequence_ids: List[List[int]],
        labels: List[int],
        tokenizer: PreTrainedTokenizerFast,
        int_to_token: Dict[int, str],
        max_length: int = 512,
        sources: Optional[List[str]] = None,
    ):
        self.sequence_ids = sequence_ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.int_to_token = int_to_token
        self.max_length = max_length
        self.sources = sources

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_ids = self.sequence_ids[idx]

        text = sequence_ids_to_string(raw_ids, self.int_to_token)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item
