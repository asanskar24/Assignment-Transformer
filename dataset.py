"""
dataset.py — Data Loading, Vocabulary & Processing
DA6401 Assignment 3: "Attention Is All You Need"

Loads the Multi30k (bentrevett/multi30k) dataset from Hugging Face,
builds vocabularies for German (src) and English (tgt) using spaCy
tokenisers, and converts sentences to integer token lists.

Special tokens:
    <unk>  idx=0
    <pad>  idx=1
    <bos>  idx=2   (also referred to as <sos> in skeleton)
    <eos>  idx=3
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy
from datasets import load_dataset


# ── Special token constants ──────────────────────────────────────────
UNK_TOKEN, UNK_IDX = '<unk>', 0
PAD_TOKEN, PAD_IDX = '<pad>', 1
BOS_TOKEN, BOS_IDX = '<bos>', 2   # beginning-of-sequence (= <sos>)
EOS_TOKEN, EOS_IDX = '<eos>', 3

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


# ════════════════════════════════════════════════════════════════════
#  VOCABULARY CLASS
# ════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple string ↔ integer vocabulary.

    Usage
    -----
    vocab = Vocabulary()
    vocab.build_from_counter(counter, min_freq=2)
    idx  = vocab['hello']       # str  → int
    tok  = vocab.get_itos()[5]  # int  → str
    """

    def __init__(self) -> None:
        self._token2idx: Dict[str, int] = {}
        self._idx2token: List[str]      = []
        # Insert special tokens first so their indices are deterministic
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    # ── private helpers ──────────────────────────────────────────────

    def _add(self, token: str) -> None:
        if token not in self._token2idx:
            self._token2idx[token] = len(self._idx2token)
            self._idx2token.append(token)

    # ── public API ───────────────────────────────────────────────────

    def build_from_counter(
        self,
        counter: Counter,
        min_freq: int = 2,
    ) -> None:
        """Add all tokens in *counter* that meet the minimum frequency."""
        for token, freq in counter.most_common():
            if freq >= min_freq:
                self._add(token)

    def __getitem__(self, token: str) -> int:
        return self._token2idx.get(token, UNK_IDX)

    def get(self, token: str, default: int = UNK_IDX) -> int:
        return self._token2idx.get(token, default)

    def get_itos(self) -> List[str]:
        """Return the index-to-string list (a copy)."""
        return list(self._idx2token)

    def __len__(self) -> int:
        return len(self._idx2token)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"


# ════════════════════════════════════════════════════════════════════
#  MULTI30K DATASET
# ════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wrapper around the bentrevett/multi30k Hugging Face dataset.

    Parameters
    ----------
    split : str
        One of 'train', 'validation', 'test'.
    min_freq : int
        Minimum token frequency for inclusion in the vocabulary.
        Only used when building vocab from the training split.
    src_vocab : Vocabulary, optional
        Pre-built source vocabulary.  If None, vocab is built from
        this split (should only be done for split='train').
    tgt_vocab : Vocabulary, optional
        Pre-built target vocabulary.
    max_len : int
        Sequences longer than this (after tokenisation) are dropped.

    Attributes
    ----------
    src_vocab, tgt_vocab : Vocabulary
    data : List[Tuple[List[int], List[int]]]
        Processed (src_ids, tgt_ids) pairs.
    """

    def __init__(
        self,
        split: str = 'train',
        min_freq: int = 2,
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        max_len: int = 150,
    ) -> None:
        self.split   = split
        self.min_freq = min_freq
        self.max_len  = max_len

        # ── 1. Load HuggingFace dataset ──────────────────────────────
        raw = load_dataset('bentrevett/multi30k', trust_remote_code=True)
        self._raw_split = raw[split]   # Arrow dataset for this split

        # ── 2. Load spaCy tokenisers ─────────────────────────────────
        try:
            self._de_nlp = spacy.load('de_core_news_sm')
        except OSError:
            raise OSError(
                "German spaCy model not found. "
                "Run: python -m spacy download de_core_news_sm"
            )
        try:
            self._en_nlp = spacy.load('en_core_web_sm')
        except OSError:
            raise OSError(
                "English spaCy model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

        # ── 3. Build or adopt vocabularies ──────────────────────────
        if src_vocab is not None and tgt_vocab is not None:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
        else:
            self.src_vocab, self.tgt_vocab = self.build_vocab()

        # ── 4. Tokenise & numericalize ───────────────────────────────
        self.data = self.process_data()

    # ── Tokenisers (public so model.infer can store them) ────────────

    def tokenize_de(self, text: str) -> List[str]:
        """Lowercase + spaCy German tokenisation."""
        return [tok.text for tok in self._de_nlp.tokenizer(text.lower())]

    def tokenize_en(self, text: str) -> List[str]:
        """Lowercase + spaCy English tokenisation."""
        return [tok.text for tok in self._en_nlp.tokenizer(text.lower())]

    # ── build_vocab ──────────────────────────────────────────────────

    def build_vocab(self) -> Tuple[Vocabulary, Vocabulary]:
        """
        Builds the vocabulary mapping for src (de) and tgt (en).

        Special tokens are always at fixed indices:
            <unk>=0, <pad>=1, <bos>=2, <eos>=3

        Returns
        -------
        src_vocab, tgt_vocab : Vocabulary
        """
        de_counter: Counter = Counter()
        en_counter: Counter = Counter()

        for example in self._raw_split:
            de_counter.update(self.tokenize_de(example['de']))
            en_counter.update(self.tokenize_en(example['en']))

        src_vocab = Vocabulary()
        tgt_vocab = Vocabulary()
        src_vocab.build_from_counter(de_counter, min_freq=self.min_freq)
        tgt_vocab.build_from_counter(en_counter, min_freq=self.min_freq)

        return src_vocab, tgt_vocab

    # ── process_data ─────────────────────────────────────────────────

    def process_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        Convert German and English sentences into integer token lists.

        Each sentence is wrapped with BOS and EOS tokens:
            [<bos>, tok_1, tok_2, ..., tok_n, <eos>]

        Pairs where either side exceeds *max_len* tokens are dropped.

        Returns
        -------
        List of (src_ids, tgt_ids) tuples.
        """
        processed = []
        for example in self._raw_split:
            de_tokens = self.tokenize_de(example['de'])
            en_tokens = self.tokenize_en(example['en'])

            # Drop over-length sequences
            if len(de_tokens) > self.max_len or len(en_tokens) > self.max_len:
                continue

            src_ids = (
                [BOS_IDX]
                + [self.src_vocab[t] for t in de_tokens]
                + [EOS_IDX]
            )
            tgt_ids = (
                [BOS_IDX]
                + [self.tgt_vocab[t] for t in en_tokens]
                + [EOS_IDX]
            )
            processed.append((src_ids, tgt_ids))

        return processed

    # ── Dataset protocol ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self.data[idx]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


# ════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTION & DATALOADER FACTORY
# ════════════════════════════════════════════════════════════════════

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a batch of (src, tgt) tensor pairs to equal length.

    Args
    ----
    batch   : list of (src_tensor, tgt_tensor) pairs
    pad_idx : padding index (default 1)

    Returns
    -------
    src_batch : [batch_size, max_src_len]
    tgt_batch : [batch_size, max_tgt_len]
    """
    src_list, tgt_list = zip(*batch)
    src_batch = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


def get_dataloaders(
    batch_size: int = 128,
    min_freq:   int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Convenience factory that creates train/val/test DataLoaders.

    Vocabularies are built ONLY from the training split, then reused
    for validation and test to avoid data leakage.

    Parameters
    ----------
    batch_size  : number of examples per batch
    min_freq    : minimum token frequency for vocab inclusion
    num_workers : DataLoader worker processes

    Returns
    -------
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Build vocab from train split
    train_ds = Multi30kDataset(split='train', min_freq=min_freq)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # Reuse those vocabs for val/test
    val_ds  = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_ds = Multi30kDataset(split='test',       src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    _collate = lambda b: collate_fn(b, pad_idx=PAD_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ════════════════════════════════════════════════════════════════════
#  QUICK SMOKE-TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading train split …")
    train_loader, val_loader, test_loader, src_v, tgt_v = get_dataloaders(batch_size=32)

    print(f"  src vocab size : {len(src_v)}")
    print(f"  tgt vocab size : {len(tgt_v)}")
    print(f"  train batches  : {len(train_loader)}")
    print(f"  val   batches  : {len(val_loader)}")
    print(f"  test  batches  : {len(test_loader)}")

    src_batch, tgt_batch = next(iter(train_loader))
    print(f"  sample src batch shape: {src_batch.shape}")
    print(f"  sample tgt batch shape: {tgt_batch.shape}")
    print("Done.")