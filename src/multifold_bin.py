"""Control experiment: the same Siamese CNN, trained on Otsu-binarised images.
Same pairs, folds, seeds and hyperparameters as the CNN baseline (multifold.py),
and the same 105x105 PIL resize used for the GNN graphs."""
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import SiameseCNN
from train import ContrastiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def binarise(img_u8):
    """Ink = 1, paper = 0, via Otsu. Removes dependence on absolute brightness."""
    _, ink = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink.astype(np.float32) / 255.0


class BinPairDataset(Dataset):
    def __init__(self, csv_path, img_size=105):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        if path not in self.cache:
            img = Image.open(path).convert("L").resize((self.img_size, self.img_size))
            self.cache[path] = torch.from_numpy(binarise(np.array(img))).unsqueeze(0)
        return self.cache[path]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self._load(row["img_a"]), self._load(row["img_b"]), row["label"]


SEEDS = [42, 1, 7, 123, 2024]
EPOCHS = 60
results = []

for fold_num, seed in enumerate(SEEDS, start=1):
    print(f"\n{'='*40}\nBINARISED-CNN FOLD {fold_num} (seed={seed})\n{'='*40}")
    torch.manual_seed(seed)

    fold_dir = f"metadata/folds/fold_{fold_num}"
    train_loader = DataLoader(BinPairDataset(f"{fold_dir}/train_pairs.csv"), batch_size=32, shuffle=True)
    test_loader = DataLoader(BinPairDataset(f"{fold_dir}/test_pairs.csv"), batch_size=32, shuffle=False)

    model = SiameseCNN(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   # same as the CNN baseline

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for a, b, label in train_loader:
            a, b, label = a.to(device), b.to(device), label.to(device).float()
            optimizer.zero_grad()
            ea, eb = model(a, b)
            loss = criterion(ea, eb, label)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - loss: {total/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"{fold_dir}/cnn_bin_model_60ep.pth")

    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for a, b, label in test_loader:
            ea, eb = model(a.to(device), b.to(device))
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
    print(f"  Binarised CNN fold {fold_num}: acc={best_acc:.4f}, threshold={best_t:.2f}, FAR={far:.4f}, FRR={frr:.4f}")
    results.append({"fold": fold_num, "seed": seed, "accuracy": best_acc,
                    "threshold": best_t, "far": far, "frr": frr})

df = pd.DataFrame(results)
print(f"\n{'='*40}\nBINARISED-CNN SUMMARY ACROSS {len(SEEDS)} FOLDS\n{'='*40}")
print(df)
print(f"\nAccuracy: mean={df.accuracy.mean():.4f}, std={df.accuracy.std():.4f}")
print(f"FAR: mean={df.far.mean():.4f}   FRR: mean={df.frr.mean():.4f}")
df.to_csv("metadata/multi_fold_cnnbin_60ep_results.csv", index=False)
