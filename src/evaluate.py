import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SignaturePairDataset
from model import SiameseCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseCNN(embedding_dim=128).to(device)
model.load_state_dict(torch.load("siamese_model.pth"))
model.eval()

test_ds = SignaturePairDataset("metadata/test_pairs.csv")
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

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

same_dists = all_dists[all_labels == 1]
diff_dists = all_dists[all_labels == 0]

print(f"Same-pair distances: mean={same_dists.mean():.4f}, std={same_dists.std():.4f}, max={same_dists.max():.4f}")
print(f"Diff-pair distances: mean={diff_dists.mean():.4f}, std={diff_dists.std():.4f}, min={diff_dists.min():.4f}")

# sweep thresholds to find best accuracy
best_acc, best_thresh = 0, 0
for thresh in np.arange(0.1, 1.5, 0.01):
    preds = (all_dists < thresh).astype(int)  # below threshold = predicted "same"
    acc = (preds == all_labels).mean()
    if acc > best_acc:
        best_acc, best_thresh = acc, thresh

print(f"Best threshold: {best_thresh:.2f} -> Accuracy: {best_acc:.4f}")

# FAR/FRR at best threshold
preds = (all_dists < best_thresh).astype(int)
false_accepts = ((preds == 1) & (all_labels == 0)).sum()  # forged wrongly accepted
false_rejects = ((preds == 0) & (all_labels == 1)).sum()  # genuine wrongly rejected
print(f"False Acceptance Rate: {false_accepts / (all_labels==0).sum():.4f}")
print(f"False Rejection Rate: {false_rejects / (all_labels==1).sum():.4f}")