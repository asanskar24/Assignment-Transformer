"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from evaluate import load as load_metric
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, BOS_IDX, EOS_IDX, PAD_IDX
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - eps
        y_smooth[other]   = eps / (vocab_size - 2)   # -2: exclude correct & pad
        y_smooth[pad]     = 0

    Implemented via KL-divergence between the smoothed distribution and
    log-softmax of model logits.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (mean over non-pad positions).
        """
        # Build smoothed target distribution
        # Start with uniform smoothing mass over all vocab entries
        smooth_val = self.smoothing / (self.vocab_size - 2)   # exclude gold & pad
        with torch.no_grad():
            dist = torch.full_like(logits, smooth_val)                 # [N, V]
            dist.scatter_(1, target.unsqueeze(1), self.confidence)     # gold gets 1-eps
            dist[:, self.pad_idx] = 0.0                                # pad gets 0
            # Zero out rows where the target IS a pad token
            pad_mask = (target == self.pad_idx)
            dist[pad_mask] = 0.0

        # KL divergence: sum(p * (log p - log q))  but p*log(p) is constant
        # so we minimise -sum(p * log_softmax(logits))
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(dist * log_probs).sum(dim=-1)   # [N]

        # Average only over non-pad tokens
        non_pad = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    phase        = "train" if is_train else "val"

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{phase}]", leave=False)
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Teacher-forcing: feed tgt[:-1], predict tgt[1:]
            tgt_input  = tgt[:, :-1]   # [batch, tgt_len-1]
            tgt_target = tgt[:, 1:]    # [batch, tgt_len-1]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX)

            # Forward pass → [batch, tgt_len-1, vocab_size]
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Flatten for loss: [batch*(tgt_len-1), vocab_size]
            batch_size, tgt_len, vocab_size = logits.shape
            logits_flat  = logits.reshape(-1, vocab_size)
            targets_flat = tgt_target.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (stabilises training)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Accumulate (weight by non-pad token count for honest averaging)
            non_pad = (targets_flat != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            # Live progress
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # W&B step-level logging (train only)
            if is_train:
                step_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
                wandb.log({
                    f"{phase}/step_loss": loss.item(),
                    "lr": step_lr,
                })

    avg_loss = total_loss / max(total_tokens, 1)
    wandb.log({f"{phase}/epoch_loss": avg_loss, "epoch": epoch_num})
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <bos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # Encode the source once
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]

        # Initialise decoder input with <bos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # Greedily pick the highest-probability token at the last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [1, 1]
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == end_symbol:
                break

    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary object with get_itos() method.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()
    bleu_metric = load_metric("bleu")

    itos        = tgt_vocab.get_itos()          # index → token string
    special_ids = {BOS_IDX, EOS_IDX, PAD_IDX}

    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            # Decode each sentence in the batch individually
            for i in range(src.size(0)):
                src_i      = src[i].unsqueeze(0)                         # [1, src_len]
                src_mask_i = make_src_mask(src_i, pad_idx=PAD_IDX)

                out = greedy_decode(
                    model, src_i, src_mask_i,
                    max_len=max_len,
                    start_symbol=BOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                # Convert prediction indices → token list (strip specials)
                pred_tokens = [
                    itos[idx] for idx in out[0].tolist()
                    if idx not in special_ids
                ]

                # Convert reference indices → token list (strip specials)
                ref_tokens = [
                    itos[idx] for idx in tgt[i].tolist()
                    if idx not in special_ids
                ]

                predictions.append(pred_tokens)
                references.append([ref_tokens])   # BLEU expects list of references

    # evaluate library expects: predictions = list of str, references = list of list of str
    # Convert token lists to strings
    pred_strings = [" ".join(p) for p in predictions]
    ref_strings  = [[" ".join(r[0])] for r in references]

    result     = bleu_metric.compute(predictions=pred_strings, references=ref_strings)
    bleu_score = result["bleu"] * 100   # convert 0-1 → 0-100
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    # Gather constructor kwargs so the autograder can rebuild the model
    model_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config":         model_config,
        },
        path,
    )
    print(f"  ✓ Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    print(f"  ✓ Checkpoint loaded ← {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    # ── Hyperparameters ──────────────────────────────────────────────
    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        warmup_steps = 4000,
        num_epochs   = 15,
        batch_size   = 128,
        min_freq     = 2,
        label_smooth = 0.1,
        max_len      = 100,
        device       = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint   = "best_checkpoint.pt",
    )

    # ── W&B ─────────────────────────────────────────────────────────
    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config
    device = cfg.device
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
    )
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Vocab sizes — src: {src_vocab_size}, tgt: {tgt_vocab_size}")

    # ── Model ────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        pad_idx        = PAD_IDX,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    wandb.log({"model/num_params": num_params})

    # ── Optimiser & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model      = cfg.d_model,
        warmup_steps = cfg.warmup_steps,
    )

    # ── Loss ─────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = tgt_vocab_size,
        pad_idx    = PAD_IDX,
        smoothing  = cfg.label_smooth,
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=cfg.checkpoint)
            wandb.log({"best_val_loss": best_val_loss, "epoch": epoch})

    # ── Final evaluation ─────────────────────────────────────────────
    # Reload best checkpoint
    load_checkpoint(cfg.checkpoint, model)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"\nTest BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()