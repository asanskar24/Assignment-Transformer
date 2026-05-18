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
    """
    Build a padding mask for the encoder (source sequence).
    Returns: Boolean mask [batch, 1, 1, src_len], True = PAD
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build combined padding + causal mask for the decoder.
    Returns: Boolean mask [batch, 1, tgt_len, tgt_len], True = masked
    """
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
    """
    Multi-Head Attention — NOT using torch.nn.MultiheadAttention.
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
    """

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

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, s, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self._merge_heads(out))


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding — registered as a buffer (not trainable).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout  = nn.Dropout(p=dropout)
        pe            = torch.zeros(max_len, d_model)
        position      = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term      = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2]   = torch.sin(position * div_term)
        pe[:, 1::2]   = torch.cos(position * div_term)
        # Registered as buffer — not a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] → same shape"""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]  (Post-LN)"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    x → [Masked Self-Attn → Add & Norm]
      → [Cross-Attn(memory) → Add & Norm]
      → [FFN → Add & Norm]   (Post-LN)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer.

    When called with no arguments — Transformer() — the model:
      1. Searches local paths for best_checkpoint.pt
      2. Falls back to downloading from Google Drive
      3. Reads model_config + weights + vocabs from the checkpoint
      4. Builds the spaCy tokenizer internally
    So model.infer(src_sentence) works with just a raw string.
    """

    # ── PASTE YOUR GOOGLE DRIVE FILE ID HERE ──────────────────────────
    GDRIVE_FILE_ID     = "1FVIDBq2q4UgjPmpbzWAiyQMZ4pNFSD1z"
    DEFAULT_CHECKPOINT = "best_checkpoint.pt"
    # ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        src_vocab_size: int   = None,
        tgt_vocab_size: int   = None,
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
        pad_idx:        int   = 1,
    ) -> None:
        super().__init__()

        _deferred_state = None
        _src_vocab      = None
        _tgt_vocab      = None

        # ── No-arg call: load everything from checkpoint ──────────────
        if src_vocab_size is None or tgt_vocab_size is None:

            # 1) Search local paths first
            _this_dir    = os.path.dirname(os.path.abspath(__file__))
            search_paths = [
                checkpoint_path or self.DEFAULT_CHECKPOINT,
                os.path.join(_this_dir, self.DEFAULT_CHECKPOINT),
                "/autograder/source/best_checkpoint.pt",
                "/autograder/submission/best_checkpoint.pt",
                "/tmp/best_checkpoint.pt",
            ]
            loaded_ckpt = None
            for p in search_paths:
                if p and os.path.exists(p):
                    loaded_ckpt = torch.load(p, map_location='cpu')
                    print(f"Loaded checkpoint from: {p}")
                    break

            # 2) Fallback: download from Google Drive
            if loaded_ckpt is None:
                local_path = "/tmp/best_checkpoint.pt"
                print(f"Downloading checkpoint from Google Drive "
                      f"(id={self.GDRIVE_FILE_ID})...")
                gdown.download(
                    id=self.GDRIVE_FILE_ID,
                    output=local_path,
                    quiet=False,
                )
                loaded_ckpt = torch.load(local_path, map_location='cpu')

            # 3) Extract architecture config
            cfg            = loaded_ckpt["model_config"]
            src_vocab_size = cfg["src_vocab_size"]
            tgt_vocab_size = cfg["tgt_vocab_size"]
            d_model        = cfg.get("d_model",   d_model)
            N              = cfg.get("N",          N)
            num_heads      = cfg.get("num_heads",  num_heads)
            d_ff           = cfg.get("d_ff",       d_ff)
            dropout        = cfg.get("dropout",    dropout)
            pad_idx        = cfg.get("pad_idx",    pad_idx)

            # 4) Store weights + vocabs for deferred loading
            _deferred_state = loaded_ckpt["model_state_dict"]
            _src_vocab      = loaded_ckpt.get("src_vocab", None)
            _tgt_vocab      = loaded_ckpt.get("tgt_vocab", None)

        self.d_model = d_model
        self.pad_idx = pad_idx

        # ── Embeddings ───────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # ── Positional Encoding ──────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # ── Encoder & Decoder ────────────────────────────────────────
        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # ── Output projection ────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # ── Xavier weight init ───────────────────────────────────────
        self._init_weights()

        # ── Load weights (no-arg path) ───────────────────────────────
        if _deferred_state is not None:
            self.load_state_dict(_deferred_state)

            # Attach vocab + tokenizer so infer() works with just a string
            # (as noted in the skeleton: "set self.src_vocab etc. after construction")
            if _src_vocab is not None and _tgt_vocab is not None:
                import spacy
                try:
                    _de_nlp = spacy.load('de_core_news_sm')
                except OSError:
                    import subprocess, sys
                    subprocess.run(
                        [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
                        check=True,
                    )
                    _de_nlp = spacy.load('de_core_news_sm')

                self.src_vocab     = _src_vocab
                self.tgt_vocab     = _tgt_vocab
                # Capture _de_nlp in closure so lambda works after __init__
                self.src_tokenizer = lambda text, nlp=_de_nlp: [
                    tok.text for tok in nlp.tokenizer(text.lower())
                ]

        # ── Explicit checkpoint_path passed ──────────────────────────
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(
                state["model_state_dict"] if "model_state_dict" in state else state
            )

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ─────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder stack.
        src: [batch, src_len] → memory: [batch, src_len, d_model]
        """
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
        """
        Run the full decoder stack and project to vocabulary logits.
        Returns: logits [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full encoder-decoder pass. Returns logits [batch, tgt_len, tgt_vocab_size]"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(
        self,
        src_sentence: str,
        src_vocab=None,
        tgt_vocab=None,
        src_tokenizer=None,
        bos_idx: int = 2,
        eos_idx: int = 3,
        max_len: int = 100,
        device:  str = 'cpu',
    ) -> str:
        """
        Translates a German sentence to English using greedy decoding.

        The autograder calls model.infer(src_sentence) with just a raw string.
        Vocab + tokenizer are loaded automatically from checkpoint in __init__
        and stored as self.src_vocab, self.tgt_vocab, self.src_tokenizer.
        """
        self.eval()

        # Use stored references if not passed explicitly
        if src_vocab     is None: src_vocab     = getattr(self, 'src_vocab',     None)
        if tgt_vocab     is None: tgt_vocab     = getattr(self, 'tgt_vocab',     None)
        if src_tokenizer is None: src_tokenizer = getattr(self, 'src_tokenizer', None)

        if src_vocab is None or tgt_vocab is None or src_tokenizer is None:
            raise ValueError(
                "src_vocab, tgt_vocab, src_tokenizer not found. "
                "Make sure the checkpoint contains 'src_vocab' and 'tgt_vocab' keys. "
                "Set self.src_vocab, self.tgt_vocab, self.src_tokenizer on the model "
                "or pass them as arguments to infer()."
            )

        with torch.no_grad():
            tokens      = src_tokenizer(src_sentence.lower())
            src_indices = (
                [src_vocab['<bos>']]
                + [src_vocab.get(t, src_vocab['<unk>']) for t in tokens]
                + [src_vocab['<eos>']]
            )
            src      = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
            src_mask = make_src_mask(src, pad_idx=self.pad_idx)
            memory   = self.encode(src, src_mask)

            tgt_indices = [bos_idx]
            for _ in range(max_len):
                tgt      = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = make_tgt_mask(tgt, pad_idx=self.pad_idx)
                logits   = self.decode(memory, src_mask, tgt, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1).item()
                tgt_indices.append(next_tok)
                if next_tok == eos_idx:
                    break

            itos    = tgt_vocab.get_itos()
            special = {'<bos>', '<eos>', '<pad>', '<unk>'}
            words   = [
                itos[idx] for idx in tgt_indices[1:]
                if idx < len(itos) and itos[idx] not in special
            ]
            return ' '.join(words)