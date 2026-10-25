import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from graphdata import GraphPairDataset
from gnn_model import SiameseGNN
from train import ContrastiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEEDS = [42, 1, 7, 123, 2024]
EPOCHS = 20
LR = 1e-3
results = []

for fold_num, seed in enumerate(SEEDS, start=1):
    print(f"\n{'='*40}\nGNN FOLD {fold_num} (seed={seed})\n{'='*40}")
    torch.manual_seed(seed)

    fold_dir = f"metadata/folds/fold_{fold_num}"      # same pairs/writers as the CNN folds
    train_ds = GraphPairDataset(f"{fold_dir}/train_pairs.csv")
    test_ds = GraphPairDataset(f"{fold_dir}/test_pairs.csv")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = SiameseGNN(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for ca, ia, cb, ib, label in train_loader:
            ca, ia, cb, ib = ca.to(device), ia.to(device), cb.to(device), ib.to(device)
            label = label.to(device).float()
            optimizer.zero_grad()
            ea, eb = model(ca, ia, cb, ib)
            loss = criterion(ea, eb, label)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - loss: {total/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"{fold_dir}/gnn_model.pth")

    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for ca, ia, cb, ib, label in test_loader:
            ea, eb = model(ca.to(device), ia.to(device), cb.to(device), ib.to(device))
            dists.extend(torch.nn.functional.pairwise_distance(ea, eb).cpu().numpy())
            labels.extend(label.numpy())
    dists, labels = np.array(dists), np.array(labels)

    best_acc, best_t = 0, 0
    for t in np.arange(0.1, 2.0, 0.01):
        acc = ((dists < t).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    preds = (dists < best_t).astype(int)
    far = ((preds == 1) & (labels == 0)).sum() / (labels == 0).sum()
    frr = ((preds == 0) & (labels == 1)).sum() / (labels == 1).sum()
    print(f"  GNN fold {fold_num}: acc={best_acc:.4f}, threshold={best_t:.2f}, FAR={far:.4f}, FRR={frr:.4f}")
    results.append({"fold": fold_num, "seed": seed, "accuracy": best_acc,
                    "threshold": best_t, "far": far, "frr": frr})

df = pd.DataFrame(results)
print(f"\n{'='*40}\nGNN SUMMARY ACROSS {len(SEEDS)} FOLDS\n{'='*40}")
print(df)
print(f"\nAccuracy: mean={df.accuracy.mean():.4f}, std={df.accuracy.std():.4f}")
print(f"FAR: mean={df.far.mean():.4f}   FRR: mean={df.frr.mean():.4f}")
df.to_csv("metadata/multi_fold_gnn_results.csv", index=False)
