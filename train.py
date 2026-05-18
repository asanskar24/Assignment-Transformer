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
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]
        Returns:
            Scalar loss value.
        """
        smooth_val = self.smoothing / (self.vocab_size - 2)
        with torch.no_grad():
            dist = torch.full_like(logits, smooth_val)
            dist.scatter_(1, target.unsqueeze(1), self.confidence)
            dist[:, self.pad_idx] = 0.0
            pad_mask = (target == self.pad_idx)
            dist[pad_mask] = 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        loss      = -(dist * log_probs).sum(dim=-1)
        non_pad   = (~pad_mask).sum().clamp(min=1)
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
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    phase        = "train" if is_train else "val"

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{phase}]", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            batch_size, tgt_len, vocab_size = logits.shape
            logits_flat  = logits.reshape(-1, vocab_size)
            targets_flat = tgt_target.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad       = (targets_flat != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if is_train:
                step_lr = optimizer.param_groups[0]["lr"]
                wandb.log({f"{phase}/step_loss": loss.item(), "lr": step_lr})

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
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask   = make_tgt_mask(ys, pad_idx=PAD_IDX)
            logits     = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys         = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
    return ys


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
    model.eval()
    bleu_metric = load_metric("bleu")
    itos        = tgt_vocab.get_itos()
    special_ids = {BOS_IDX, EOS_IDX, PAD_IDX}
    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            for i in range(src.size(0)):
                src_i      = src[i].unsqueeze(0)
                src_mask_i = make_src_mask(src_i, pad_idx=PAD_IDX)
                out = greedy_decode(
                    model, src_i, src_mask_i,
                    max_len=max_len,
                    start_symbol=BOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )
                pred_tokens = [itos[idx] for idx in out[0].tolist() if idx not in special_ids]
                ref_tokens  = [itos[idx] for idx in tgt[i].tolist()  if idx not in special_ids]
                predictions.append(" ".join(pred_tokens))
                references.append([" ".join(ref_tokens)])

    result     = bleu_metric.compute(predictions=predictions, references=references)
    bleu_score = result["bleu"] * 100
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
    Save model + optimizer + scheduler + vocab state to disk.
    Vocabs are read from model.src_vocab / model.tgt_vocab if attached.
    """
    model_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
        "pad_idx":        model.pad_idx,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config":         model_config,
            # Save vocabs so infer() works with no args after loading
            "src_vocab":            getattr(model, 'src_vocab',  None),
            "tgt_vocab":            getattr(model, 'tgt_vocab',  None),
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
        device       = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint   = "best_checkpoint.pt",
    )

    wandb.init(project="da6401-a3", config=config)
    cfg    = wandb.config
    device = cfg.device

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size, min_freq=cfg.min_freq,
    )

    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        pad_idx        = PAD_IDX,
    ).to(device)

    # Attach vocabs to model so save_checkpoint picks them up
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(vocab_size=len(tgt_vocab), pad_idx=PAD_IDX, smoothing=cfg.label_smooth)

    best_val_loss = float("inf")
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_loader, model, loss_fn, None, None,
                               epoch_num=epoch, is_train=False, device=device)
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=cfg.checkpoint)
            print(f"  → New best (val {best_val_loss:.4f})")

    load_checkpoint(cfg.checkpoint, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"\nTest BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()