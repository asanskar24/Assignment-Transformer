"""
model.py — Transformer Architecture Implementation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import subprocess
import sys
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
    Positions where mask is True are set to -inf before softmax.
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    return torch.matmul(attn_w, V), attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """[batch, src_len] → [batch, 1, 1, src_len], True=PAD"""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """[batch, tgt_len] → [batch, 1, tgt_len, tgt_len], True=masked"""
    tgt_len     = tgt.size(1)
    device      = tgt.device
    pad_mask    = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention — NOT using torch.nn.MultiheadAttention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q       = nn.Linear(d_model, d_model, bias=False)
        self.W_k       = nn.Linear(d_model, d_model, bias=False)
        self.W_v       = nn.Linear(d_model, d_model, bias=False)
        self.W_o       = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(p=dropout)

    def _split_heads(self, x):
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        b, _, s, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self._merge_heads(out))


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal PE — registered as buffer (not a trainable parameter)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW1 + b1)W2 + b2"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """Post-LN: x → [Self-Attn → Add&Norm] → [FFN → Add&Norm]"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """Post-LN: MaskedSelfAttn → CrossAttn → FFN, each with Add&Norm"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

def _load_spacy_de():
    """Load German spaCy model, auto-downloading if missing."""
    import spacy
    try:
        return spacy.load('de_core_news_sm')
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
            check=True,
        )
        return spacy.load('de_core_news_sm')


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German→English NMT.

    Per assignment instructions:
      - Transformer() with NO args must work end-to-end.
      - Weights downloaded via gdown inside __init__.
      - Vocab + tokenizer loaded inside __init__.
      - model.infer(german_sentence) returns English string.
    """

    # ── YOUR GOOGLE DRIVE FILE ID ──────────────────────────────────────
    GDRIVE_FILE_ID     = "1FVIDBq2q4UgjPmpbzWAiyQMZ4pNFSD1z"
    CKPT_LOCAL_PATH    = "/tmp/best_checkpoint.pt"
    # ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        src_vocab_size: int   = None,
        tgt_vocab_size: int   = None,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
        pad_idx:        int   = 1,
    ) -> None:
        super().__init__()

        # ── Step 1: Download checkpoint from Google Drive ─────────────
        # Always download to ensure we have the latest weights.
        # This happens inside __init__ as required by the assignment.
        if not os.path.exists(self.CKPT_LOCAL_PATH):
            print(f"Downloading checkpoint from Google Drive "
                  f"(id={self.GDRIVE_FILE_ID})...")
            gdown.download(
                id=self.GDRIVE_FILE_ID,
                output=self.CKPT_LOCAL_PATH,
                quiet=False,
            )

        checkpoint = torch.load(self.CKPT_LOCAL_PATH, map_location='cpu')

        # ── Step 2: Read architecture config from checkpoint ──────────
        cfg           = checkpoint["model_config"]
        src_vocab_size = cfg["src_vocab_size"]
        tgt_vocab_size = cfg["tgt_vocab_size"]
        d_model        = cfg.get("d_model",   d_model)
        N              = cfg.get("N",          N)
        num_heads      = cfg.get("num_heads",  num_heads)
        d_ff           = cfg.get("d_ff",       d_ff)
        dropout        = cfg.get("dropout",    dropout)
        pad_idx        = cfg.get("pad_idx",    pad_idx)

        self.d_model = d_model
        self.pad_idx = pad_idx

        # ── Step 3: Build model architecture ─────────────────────────
        self.src_embed   = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed   = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc     = PositionalEncoding(d_model, dropout)
        enc_layer        = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer        = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Xavier init then overwrite with trained weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ── Step 4: Load trained weights ──────────────────────────────
        self.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully.")

        # ── Step 5: Load vocab and build tokenizer ────────────────────
        # Stored inside checkpoint by train.py's save_checkpoint()
        src_vocab = checkpoint.get("src_vocab", None)
        tgt_vocab = checkpoint.get("tgt_vocab", None)

        if src_vocab is None or tgt_vocab is None:
            raise RuntimeError(
                "Checkpoint does not contain 'src_vocab' or 'tgt_vocab'. "
                "Please re-save your checkpoint using the updated train.py "
                "which calls save_checkpoint() after attaching "
                "model.src_vocab and model.tgt_vocab."
            )

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Build German tokenizer using spaCy (auto-downloads if needed)
        _de_nlp = _load_spacy_de()
        self.src_tokenizer = lambda text, nlp=_de_nlp: [
            tok.text for tok in nlp.tokenizer(text.lower())
        ]
        print("Vocab and tokenizer loaded successfully.")

    # ── AUTOGRADER HOOKS ─────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """src [batch, src_len] → memory [batch, src_len, d_model]"""
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits [batch, tgt_len, tgt_vocab_size]"""
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Full encoder-decoder pass → logits [batch, tgt_len, tgt_vocab_size]"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(
        self,
        src_sentence: str,
        bos_idx: int = 2,
        eos_idx: int = 3,
        max_len: int = 100,
        device:  str = 'cpu',
    ) -> str:
        """
        End-to-end NMT inference:
          1. Tokenize German sentence with spaCy
          2. Numericalize with src_vocab
          3. Encode with Transformer encoder
          4. Greedy autoregressive decoding
          5. Detokenize and return English string

        Called by autograder as: model.infer(german_sentence)
        All args have defaults — no extra arguments needed.
        """
        self.eval()

        with torch.no_grad():
            # Tokenize + numericalize source
            tokens      = self.src_tokenizer(src_sentence)
            src_indices = (
                [self.src_vocab['<bos>']]
                + [self.src_vocab.get(t, self.src_vocab['<unk>']) for t in tokens]
                + [self.src_vocab['<eos>']]
            )
            src      = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
            src_mask = make_src_mask(src, pad_idx=self.pad_idx)

            # Encode once
            memory = self.encode(src, src_mask)

            # Greedy autoregressive decoding
            tgt_indices = [bos_idx]
            for _ in range(max_len):
                tgt      = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = make_tgt_mask(tgt, pad_idx=self.pad_idx)
                logits   = self.decode(memory, src_mask, tgt, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1).item()
                tgt_indices.append(next_tok)
                if next_tok == eos_idx:
                    break

            # Detokenize — strip special tokens
            itos    = self.tgt_vocab.get_itos()
            special = {'<bos>', '<eos>', '<pad>', '<unk>'}
            words   = [
                itos[idx] for idx in tgt_indices[1:]
                if idx < len(itos) and itos[idx] not in special
            ]
            return ' '.join(words)