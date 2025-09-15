import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

# ---------------- Custom Dataset Example (added) ----------------
# Sample sentences
sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom dataset
# class CustomDataset(Dataset):
#     def __init__(self, sentences):
#         self.sentences = sentences
#
#     def __len__(self):
#         return len(self.sentences)
#
#     def __getitem__(self, idx):
#         return self.sentences[idx]
#
# # Create an instance of your custom dataset
# custom_dataset = CustomDataset(sentences)
#
# # Define batch size
# batch_size = 2
#
# # Create a DataLoader
# dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
#
# # Iterate through the DataLoader
# for batch in dataloader:
#     print(batch)


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices, dtype=torch.long)

def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary (add special tokens optionally if needed later)
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create dataset instance
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(len(sentences)):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")

# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

# Define batch size
batch_size = 2

# Create a data loader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)

print("\nbatch prints:")
# Iterate through the data loader
for batch in dataloader:
    print(batch)

# Iterate through the data loader
for batch in dataloader:
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)
