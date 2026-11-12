"""PyTorch Training Pipeline for Neural SpatialDepthGate in FastReasonSeg-3D.

Trains a geometric cross-attention projection head on 3D spatial-depth referring tuples.
Computes Cross-Entropy loss over candidate selections, validates on held-out scenes,
saves checkpoint weights, and generates loss convergence curves.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class NeuralSpatialDepthGate(nn.Module):
    """Neural Cross-Attention Gating Head for 3D Geometric Disambiguation."""
    def __init__(self, spatial_feat_dim: int = 8, hidden_dim: int = 64, text_vocab_size: int = 128):
        super().__init__()
        # Spatial feature vector: [xc, yc, z_rel, w, h, area, depth_std, det_conf]
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Pairwise relational MLP (Δx, Δy, Δz, d3d)
        self.relation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        # Text query projection (bag of words embedding + linear projection)
        self.text_embedding = nn.EmbeddingBag(text_vocab_size, hidden_dim, mode="mean")
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Scoring projection
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_tokens: torch.Tensor, spatial_feats: torch.Tensor) -> torch.Tensor:
        """
        query_tokens: [B, num_tokens]
        spatial_feats: [B, num_candidates, spatial_feat_dim]
        returns: logits [B, num_candidates]
        """
        B, K, _ = spatial_feats.shape
        # Encode spatial features
        spatial_emb = self.spatial_encoder(spatial_feats) # [B, K, hidden_dim]

        # Encode query
        q_emb = self.query_proj(self.text_embedding(query_tokens)) # [B, hidden_dim]
        q_expanded = q_emb.unsqueeze(1).expand(-1, K, -1) # [B, K, hidden_dim]

        # Cross fusion
        fused = torch.cat([spatial_emb, q_expanded], dim=-1) # [B, K, hidden_dim * 2]
        logits = self.score_head(fused).squeeze(-1) # [B, K]
        return logits


def generate_synthetic_training_dataset(num_samples: int = 300):
    """Generates synthetic multi-object scenes with 3D spatial grounding for training."""
    dataset = []
    # Token dictionary
    vocab = {
        "front": 1, "foreground": 2, "closest": 3, "near": 4, "closer": 5,
        "back": 6, "background": 7, "behind": 8, "farthest": 9, "rear": 10,
        "left": 11, "leftmost": 12, "right": 13, "rightmost": 14,
        "top": 15, "upper": 16, "above": 17, "bottom": 18, "lower": 19, "below": 20,
        "vehicle": 21, "car": 22, "person": 23, "box": 24, "object": 25,
    }

    for _ in range(num_samples):
        num_candidates = random.randint(3, 5)
        # Generate random spatial coordinates
        # xc, yc, z_rel, w, h, area, depth_std, det_conf
        candidates = []
        for _ in range(num_candidates):
            xc = random.uniform(0.1, 0.9)
            yc = random.uniform(0.1, 0.9)
            z_rel = random.uniform(0.05, 0.95)
            w = random.uniform(0.1, 0.3)
            h = random.uniform(0.1, 0.3)
            area = w * h
            depth_std = random.uniform(0.05, 0.2)
            conf = random.uniform(0.7, 0.98)
            candidates.append([xc, yc, z_rel, w, h, area, depth_std, conf])

        # Pick a target rule
        rule_type = random.choice(["closest_front", "farthest_back", "leftmost", "rightmost", "upper", "lower"])
        if rule_type == "closest_front":
            target_idx = min(range(num_candidates), key=lambda i: candidates[i][2])
            tokens = [vocab["closest"], vocab["object"], vocab["front"], vocab["foreground"]]
        elif rule_type == "farthest_back":
            target_idx = max(range(num_candidates), key=lambda i: candidates[i][2])
            tokens = [vocab["farthest"], vocab["object"], vocab["behind"], vocab["background"]]
        elif rule_type == "leftmost":
            target_idx = min(range(num_candidates), key=lambda i: candidates[i][0])
            tokens = [vocab["leftmost"], vocab["object"], vocab["left"]]
        elif rule_type == "rightmost":
            target_idx = max(range(num_candidates), key=lambda i: candidates[i][0])
            tokens = [vocab["rightmost"], vocab["object"], vocab["right"]]
        elif rule_type == "upper":
            target_idx = min(range(num_candidates), key=lambda i: candidates[i][1])
            tokens = [vocab["upper"], vocab["object"], vocab["above"], vocab["top"]]
        else: # lower
            target_idx = max(range(num_candidates), key=lambda i: candidates[i][1])
            tokens = [vocab["lower"], vocab["object"], vocab["below"], vocab["bottom"]]

        # Pad candidates to 5
        while len(candidates) < 5:
            candidates.append([0.0] * 8)

        dataset.append({
            "tokens": tokens,
            "candidates": candidates,
            "target_idx": target_idx,
            "num_candidates": num_candidates,
        })

    return dataset, vocab


def train_model():
    print("=" * 60)
    print("STARTING PYTORCH TRAINING: Neural SpatialDepthGate Head")
    print("=" * 60)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1e9 else "cpu")
    print(f"Training Compute Device: {device}")

    # 2. Data Preparation
    full_dataset, vocab = generate_synthetic_training_dataset(num_samples=400)
    split_idx = int(len(full_dataset) * 0.8)
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]
    print(f"Training Samples: {len(train_data)} | Validation Samples: {len(val_data)}")

    # 3. Model, Optimizer, Loss
    model = NeuralSpatialDepthGate(spatial_feat_dim=8, hidden_dim=64, text_vocab_size=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def collate_batch(batch_samples):
        max_t = max(len(s["tokens"]) for s in batch_samples)
        token_tensors = []
        for s in batch_samples:
            t = s["tokens"] + [0] * (max_t - len(s["tokens"]))
            token_tensors.append(t)

        tokens_tensor = torch.tensor(token_tensors, dtype=torch.long, device=device)
        feats_tensor = torch.tensor([s["candidates"] for s in batch_samples], dtype=torch.float32, device=device)
        targets_tensor = torch.tensor([s["target_idx"] for s in batch_samples], dtype=torch.long, device=device)
        return tokens_tensor, feats_tensor, targets_tensor

    epochs = 25
    batch_size = 16
    train_losses = []
    val_accuracies = []

    print("\nTraining Progress:")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            tokens_b, feats_b, targets_b = collate_batch(batch)

            optimizer.zero_grad()
            logits = model(tokens_b, feats_b) # [B, 5]
            loss = criterion(logits, targets_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total_val = len(val_data)
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                tokens_b, feats_b, targets_b = collate_batch(batch)
                logits = model(tokens_b, feats_b)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets_b).sum().item()

        val_acc = (correct / total_val) * 100.0
        val_accuracies.append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.1f}%")

    print("-" * 60)
    print(f"Training Complete! Final Loss: {train_losses[-1]:.4f} | Peak Val Accuracy: {max(val_accuracies):.1f}%")

    # 4. Save Checkpoint
    checkpoint_path = MODELS_DIR / "neural_spatial_depth_gate.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "epochs": epochs,
        "final_loss": train_losses[-1],
        "final_val_acc": val_accuracies[-1],
    }, checkpoint_path)
    print(f"Saved Checkpoint: {checkpoint_path}")

    # 5. Plot Convergence Curve
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    color = "#ef4444"
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", color=color, fontweight="bold")
    line1 = ax1.plot(range(1, epochs + 1), train_losses, color=color, linewidth=2.2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, max(train_losses) * 1.1)

    ax2 = ax1.twinx()
    color2 = "#10b981"
    ax2.set_ylabel("Validation Accuracy (%)", color=color2, fontweight="bold")
    line2 = ax2.plot(range(1, epochs + 1), val_accuracies, color=color2, linewidth=2.2, linestyle="--", label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(40, 105)

    plt.title("Neural SpatialDepthGate: Training Convergence & Accuracy", fontweight="bold", pad=12)
    fig.tight_layout()
    curve_path = FIGURES_DIR / "training_loss_convergence.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved Convergence Curve: {curve_path}")

    # 6. Save JSON metrics
    metrics_path = RESULTS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "epochs": epochs,
            "train_losses": [round(l, 4) for l in train_losses],
            "val_accuracies": [round(a, 2) for a in val_accuracies],
            "final_train_loss": round(train_losses[-1], 4),
            "final_val_accuracy_percent": round(val_accuracies[-1], 2),
            "device": str(device),
        }, f, indent=2)
    print(f"Saved Metrics Log: {metrics_path}")


if __name__ == "__main__":
    train_model()
