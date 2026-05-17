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

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Compute raw scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask: set masked positions to -inf so softmax gives ~0
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Softmax over key dimension
    attn_w = F.softmax(scores, dim=-1)

    # Handle all-masked rows (NaN → 0)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # [batch, src_len] → [batch, 1, 1, src_len]
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
    device  = tgt.device

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask: upper triangle is True (future positions blocked)
    # shape: [1, 1, tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # Combine: mask out if EITHER is True
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Projection matrices for Q, K, V and the output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, d_model] → [batch, heads, seq, d_k]"""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch, heads, seq, d_k]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, heads, seq, d_k] → [batch, seq, d_model]"""
        batch, _, seq, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        # Linear projections → split into heads
        Q = self._split_heads(self.W_q(query))  # [batch, heads, seq_q, d_k]
        K = self._split_heads(self.W_k(key))    # [batch, heads, seq_k, d_k]
        V = self._split_heads(self.W_v(value))  # [batch, heads, seq_k, d_k]

        # Scaled dot-product attention over all heads in parallel
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: [batch, heads, seq_q, d_k]

        # Merge heads and project
        out = self.W_o(self._merge_heads(attn_out))  # [batch, seq_q, d_model]
        return out


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute PE table: [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Compute division term: 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Uses Post-LayerNorm (as in the original paper):
        x = LayerNorm(x + Sublayer(x))

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Sub-layer 1: Self-attention + Add & Norm (Post-LN)
        _x = self.self_attn(x, x, x, src_mask)
        x  = self.norm1(x + self.dropout(_x))

        # Sub-layer 2: FFN + Add & Norm (Post-LN)
        _x = self.ffn(x)
        x  = self.norm2(x + self.dropout(_x))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Uses Post-LayerNorm (original paper).

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Sub-layer 1: Masked self-attention + Add & Norm
        _x = self.self_attn(x, x, x, tgt_mask)
        x  = self.norm1(x + self.dropout(_x))

        # Sub-layer 2: Cross-attention over encoder memory + Add & Norm
        _x = self.cross_attn(x, memory, memory, src_mask)
        x  = self.norm2(x + self.dropout(_x))

        # Sub-layer 3: FFN + Add & Norm
        _x = self.ffn(x)
        x  = self.norm3(x + self.dropout(_x))
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
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
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
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Design notes
    ────────────
    • Post-LayerNorm chosen (matches original paper §5.4 and ablation
      stability at moderate d_model/N; justified vs Pre-LN in report).
    • Weight tying between src/tgt embeddings and the output projection
      is NOT applied (different vocab sizes for de→en).
    • Embedding scale: multiply by √d_model as per paper §3.4.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
        checkpoint_path(str)  : If given, download weights from Google Drive.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
        pad_idx:   int   = 1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # ── Embeddings ──────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # ── Positional Encoding ──────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # ── Encoder & Decoder Stacks ─────────────────────────────────
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # ── Output projection ────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # ── Weight initialisation (Xavier uniform, §3 of paper) ──────
        self._init_weights()

        # ── Optional checkpoint loading ──────────────────────────────
        if checkpoint_path is not None:
            gdown.download(id="1EMG2mK3ZsLICTzNdtlN9eB_Hl1o5jmPr", output=checkpoint_path, quiet=False)
            state = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(state)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ───────────────────────────────────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Scale embeddings by √d_model (paper §3.4)
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

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
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
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(
        self,
        src_sentence: str,
        src_vocab=None,
        tgt_vocab=None,
        src_tokenizer=None,
        bos_idx: int = 2,
        eos_idx: int = 3,
        max_len: int = 100,
        device: str = 'cpu',
    ) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence : The raw German text.
            src_vocab    : Source vocab object with __getitem__ (token → idx).
            tgt_vocab    : Target vocab object with get_itos() (idx → token).
            src_tokenizer: Callable that tokenises raw German text into a list of strings.
            bos_idx      : Index of <bos> token in the target vocab.
            eos_idx      : Index of <eos> token in the target vocab.
            max_len      : Maximum number of tokens to generate.
            device       : Device string ('cpu' or 'cuda').

        Returns:
            The fully translated English string, detokenized and clean.

        Note:
            The dataset.py module should pass the required vocab/tokenizer
            objects when calling this method.  The signature accepts them as
            keyword args so the autograder can call infer(src_sentence) with
            just a raw string if the model stores these references internally
            (set self.src_vocab etc. after construction).
        """
        self.eval()

        # Allow model to store vocab/tokenizer references for convenience
        if src_vocab is None:
            src_vocab = getattr(self, 'src_vocab', None)
        if tgt_vocab is None:
            tgt_vocab = getattr(self, 'tgt_vocab', None)
        if src_tokenizer is None:
            src_tokenizer = getattr(self, 'src_tokenizer', None)

        if src_vocab is None or tgt_vocab is None or src_tokenizer is None:
            raise ValueError(
                "Provide src_vocab, tgt_vocab, and src_tokenizer either as "
                "arguments or by setting self.src_vocab / self.tgt_vocab / "
                "self.src_tokenizer on the model instance."
            )

        with torch.no_grad():
            # Tokenise & numericalize the source sentence
            tokens = src_tokenizer(src_sentence.lower())
            src_indices = [src_vocab['<bos>']] + \
                          [src_vocab.get(t, src_vocab['<unk>']) for t in tokens] + \
                          [src_vocab['<eos>']]
            src = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
            src_mask = make_src_mask(src, pad_idx=self.pad_idx)

            # Encode once
            memory = self.encode(src, src_mask)

            # Greedy decode
            tgt_indices = [bos_idx]
            for _ in range(max_len):
                tgt = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = make_tgt_mask(tgt, pad_idx=self.pad_idx)

                logits = self.decode(memory, src_mask, tgt, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                tgt_indices.append(next_token)

                if next_token == eos_idx:
                    break

            # Convert indices → tokens, strip special tokens
            itos = tgt_vocab.get_itos()  # index-to-string list
            special = {'<bos>', '<eos>', '<pad>', '<unk>'}
            words = [
                itos[idx] for idx in tgt_indices[1:]   # skip <bos>
                if itos[idx] not in special and idx != eos_idx
            ]
            return ' '.join(words)