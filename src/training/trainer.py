"""Shared Trainer class for all models.

Handles training loop, evaluation (windowed + full-signal), checkpointing,
and logging. Does NOT own W&B init/finish — the script handles that.
"""

import json
import os
import time
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import get_loss_fn
from src.evaluation.metrics import pearson_r_per_channel, smooth_predictions
from src.utils.wandb_utils import log_epoch, log_summary


class Trainer:
    """Unified trainer for ECoG decoding models.

    Parameters
    ----------
    model : nn.Module
    config : dict
        Full YAML config.
    dataset_info : dict
        Return value from build_data() — contains train/val/test datasets
        and metadata.
    device : torch.device
    exp_dir : str
        Path to save checkpoints and results.
    """

    def __init__(self, model, config, dataset_info, device, exp_dir):
        self.model = model.to(device)
        self.config = config
        self.ds = dataset_info
        self.device = device
        self.exp_dir = exp_dir

        train_cfg = config["training"]
        self.loss_fn = get_loss_fn(train_cfg["loss"])
        self.grad_clip = train_cfg.get("grad_clip_max_norm", 1.0)
        self.n_targets = config["data"]["n_targets"]

        # Compute total stride for time alignment in full-signal eval
        strides = config["model"].get("strides", [1])
        self.total_stride = reduce(lambda a, b: a * b, strides, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

        # Scheduler
        sched_type = train_cfg.get("scheduler", "reduce_on_plateau")
        if sched_type == "none":
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min",
                patience=train_cfg.get("scheduler_patience", 5),
                factor=train_cfg.get("scheduler_factor", 0.5),
            )

        # DataLoaders
        batch_size = train_cfg["batch_size"]
        self.train_loader = DataLoader(
            self.ds["train"], batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.ds["val"], batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.ds["test"], batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # Eval config
        eval_cfg = config.get("evaluation", {})
        self.smooth_sigma_val = eval_cfg.get("smooth_sigma_val", 0)
        self.smooth_sigma_test = eval_cfg.get("smooth_sigma_test", 0)

    def train_one_epoch(self):
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        n_samples = 0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        return total_loss / n_samples

    @torch.no_grad()
    def evaluate_windowed(self, loader):
        """Evaluate on a DataLoader. Returns (avg_loss, r_per_channel).

        Concatenates windowed predictions, flattens, and computes correlation.
        """
        self.model.eval()
        total_loss = 0
        all_pred, all_target = [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            all_pred.append(pred.cpu().numpy())
            all_target.append(y_batch.cpu().numpy())

        n = sum(p.shape[0] for p in all_pred)
        avg_loss = total_loss / n

        # Concat and flatten: (N, C, T) → (C, N*T)
        pred_cat = np.concatenate(all_pred, axis=0)
        target_cat = np.concatenate(all_target, axis=0)
        n_ch = pred_cat.shape[1]
        pred_flat = pred_cat.transpose(1, 0, 2).reshape(n_ch, -1)
        target_flat = target_cat.transpose(1, 0, 2).reshape(n_ch, -1)

        r_per_ch = pearson_r_per_channel(pred_flat, target_flat)
        return avg_loss, r_per_ch

    @torch.no_grad()
    def evaluate_fullsig(self, dataset, smooth_sigma=0):
        """Evaluate on full continuous signal (no windowing).

        Uses dataset.get_full_signal() which works for both LomtevDataset
        and FingerFlexDataset.

        Returns
        -------
        r_per_channel : np.ndarray, shape (n_targets,)
        """
        self.model.eval()

        full_input, target = dataset.get_full_signal()
        # full_input: tensor (for Lomtev: (C,W,T), for raw: (C,T))
        # target: np.ndarray (n_targets, T)

        inp = full_input.unsqueeze(0).to(self.device)  # add batch dim

        # Align time to total_stride
        T = inp.shape[-1]
        T_aligned = (T // self.total_stride) * self.total_stride
        inp = inp[..., :T_aligned]
        target = target[..., :T_aligned]

        pred = self.model(inp).squeeze(0).cpu().numpy()  # (n_targets, T_aligned)

        if smooth_sigma > 0:
            pred = smooth_predictions(pred, smooth_sigma)

        return pearson_r_per_channel(pred, target)

    def fit(self):
        """Full training loop with early stopping. Returns best metrics dict."""
        train_cfg = self.config["training"]
        n_epochs = train_cfg["epochs"]
        patience = train_cfg.get("early_stopping_patience", 15)

        best_val_loss = float("inf")
        best_val_r = -float("inf")
        patience_counter = 0

        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Val r (avg)':>10} | {'Val r (per ch)':>30} | "
              f"{'LR':>10} | {'Time':>6}")
        print("-" * 105)

        for epoch in range(1, n_epochs + 1):
            t_start = time.time()

            train_loss = self.train_one_epoch()
            val_loss, _ = self.evaluate_windowed(self.val_loader)
            val_r = self.evaluate_fullsig(
                self.ds["val"], smooth_sigma=self.smooth_sigma_val
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]["lr"]
            r_avg = val_r.mean()
            r_str = " ".join(f"{r:.3f}" for r in val_r)

            print(f"{epoch:>5} | {train_loss:>10.6f} | {val_loss:>10.6f} | "
                  f"{r_avg:>10.4f} | {r_str:>30} | {lr:>10.2e} | {elapsed:>5.1f}s")

            # W&B logging
            epoch_metrics = {
                "train/loss": train_loss, "val/loss": val_loss,
                "val/r_avg": float(r_avg), "train/lr": lr,
                "train/epoch_time": elapsed,
            }
            for i, r in enumerate(val_r):
                epoch_metrics[f"val/r_ch_{i}"] = float(r)
            log_epoch(epoch_metrics, step=epoch)

            # Save best model (select by max correlation)
            if r_avg > best_val_r:
                best_val_r = r_avg
                best_val_loss = val_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.exp_dir, "best_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        return {"best_val_r": float(best_val_r), "best_val_loss": float(best_val_loss)}

    def test(self):
        """Evaluate best model on test set. Saves JSON + logs W&B summary.

        Returns
        -------
        dict with test results.
        """
        print("\n" + "=" * 60)
        print("TEST EVALUATION (best val model)")
        print("=" * 60)

        ckpt_path = os.path.join(self.exp_dir, "best_model.pt")
        self.model.load_state_dict(
            torch.load(ckpt_path, weights_only=True, map_location=self.device)
        )

        test_r = self.evaluate_fullsig(
            self.ds["test"], smooth_sigma=self.smooth_sigma_test
        )
        test_loss, _ = self.evaluate_windowed(self.test_loader)

        print(f"  Test loss: {test_loss:.6f}")
        print(f"  Test r (per ch): {' '.join(f'{r:.4f}' for r in test_r)}")
        print(f"  Test r (average): {test_r.mean():.4f}")

        n_params = sum(p.numel() for p in self.model.parameters())

        # W&B summary
        summary = {
            "test/loss": float(test_loss),
            "test/r_avg": float(test_r.mean()),
            "model/n_params": n_params,
            "model/n_channels": self.ds["n_channels"],
            "data/n_train": len(self.ds["train"]),
            "data/n_val": len(self.ds["val"]),
            "data/n_test": len(self.ds["test"]),
        }
        for i, r in enumerate(test_r):
            summary[f"test/r_ch_{i}"] = float(r)
        log_summary(summary)

        # Save JSON
        results = {
            "test_loss": float(test_loss),
            "test_r_per_channel": [float(r) for r in test_r],
            "test_r_avg": float(test_r.mean()),
            "n_params": n_params,
        }
        with open(os.path.join(self.exp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results
