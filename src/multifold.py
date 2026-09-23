import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from bpairs import build_pairs
from dataset import SignaturePairDataset
from model import SiameseCNN
from train import ContrastiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("metadata/cedar_metadata.csv")

SEEDS = [42, 1, 7, 123, 2024]
EPOCHS = 20
results = []

os.makedirs("metadata/folds", exist_ok=True)

for fold_num, seed in enumerate(SEEDS, start=1):
    print(f"\n{'='*40}\nFOLD {fold_num} (seed={seed})\n{'='*40}")
    torch.manual_seed(seed)  # REPRODUCIBILITY

    # --- split (same logic as split.py, looped per seed) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=10/55, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["writer_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df.writer_id) & set(test_df.writer_id)
    assert len(overlap) == 0, f"Writer leakage detected in fold {fold_num}: {overlap}"

        # --- build pairs (reusing existing function) ---
    train_pairs = build_pairs(train_df, pairs_per_writer=40)
    test_pairs = build_pairs(test_df, pairs_per_writer=40)

    fold_dir = f"metadata/folds/fold_{fold_num}"
    os.makedirs(fold_dir, exist_ok=True)
    train_pairs.to_csv(f"{fold_dir}/train_pairs.csv", index=False)
    test_pairs.to_csv(f"{fold_dir}/test_pairs.csv", index=False)

    # --- train ---
    train_ds = SignaturePairDataset(f"{fold_dir}/train_pairs.csv", augment=False)
    test_ds = SignaturePairDataset(f"{fold_dir}/test_pairs.csv", augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = SiameseCNN(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for img_a, img_b, label in train_loader:
            img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device).float()
            optimizer.zero_grad()
            emb_a, emb_b = model(img_a, img_b)
            loss = criterion(emb_a, emb_b, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"metadata/folds/fold_{fold_num}/siamese_model.pth")
    # --- evaluate ---
    model.eval()
    all_dists, all_labels = [], []
    with torch.no_grad():
        for img_a, img_b, label in test_loader:
            img_a, img_b = img_a.to(device), img_b.to(device)
            emb_a, emb_b = model(img_a, img_b)
            dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(label.numpy())

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    best_acc, best_thresh = 0, 0
    for thresh in np.arange(0.1, 2.0, 0.01):
        preds = (all_dists < thresh).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    preds = (all_dists < best_thresh).astype(int)
    far = ((preds == 1) & (all_labels == 0)).sum() / (all_labels == 0).sum()
    frr = ((preds == 0) & (all_labels == 1)).sum() / (all_labels == 1).sum()

    print(f"  Fold {fold_num} result: acc={best_acc:.4f}, threshold={best_thresh:.2f}, FAR={far:.4f}, FRR={frr:.4f}")
    results.append({"fold": fold_num, "seed": seed, "accuracy": best_acc, "threshold": best_thresh, "far": far, "frr": frr})

# --- summary across all folds ---
results_df = pd.DataFrame(results)
print(f"\n{'='*40}\nSUMMARY ACROSS {len(SEEDS)} FOLDS\n{'='*40}")
print(results_df)
print(f"\nAccuracy: mean={results_df.accuracy.mean():.4f}, std={results_df.accuracy.std():.4f}")
print(f"FAR: mean={results_df.far.mean():.4f}, std={results_df.far.std():.4f}")
print(f"FRR: mean={results_df.frr.mean():.4f}, std={results_df.frr.std():.4f}")

results_df.to_csv("metadata/multi_fold_results.csv", index=False)
print("\nSaved to metadata/multi_fold_results.csv")