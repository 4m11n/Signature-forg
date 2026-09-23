import torch
import numpy as np
import pandas as pd
import albumentations as A
from model import SiameseCNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 105

CORRUPTIONS = {
    "clean": [A.NoOp()],
    "compression": [A.ImageCompression(quality_range=(q, q), p=1.0) for q in [70, 50, 30, 15, 5]],
    "blur": [A.GaussianBlur(blur_limit=(k, k), p=1.0) for k in [3, 5, 9, 13, 17]],
    "brightness": [A.RandomBrightnessContrast(brightness_limit=b, contrast_limit=0, p=1.0) for b in [0.1, 0.2, 0.3, 0.4, 0.5]],
    "perspective": [A.Perspective(scale=(s, s), p=1.0) for s in [0.02, 0.05, 0.08, 0.12, 0.16]],
}

def load_and_corrupt(path, transform):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img)
    augmented = transform(image=img_np)["image"]
    tensor = torch.from_numpy(augmented).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(device)

def embed(model, path, transform):
    with torch.no_grad():
        return model.forward_once(load_and_corrupt(path, transform))

def evaluate_fold_corruption(fold_num):
    model = SiameseCNN(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(f"metadata/folds/fold_{fold_num}/siamese_model.pth"))
    model.eval()

    test_pairs = pd.read_csv(f"metadata/folds/fold_{fold_num}/test_pairs.csv")
    results = []

    for corruption_name, severities in CORRUPTIONS.items():
        for severity_idx, aug in enumerate(severities):
            transform = A.Compose([aug])
            all_dists, all_labels = [], []

            for _, row in test_pairs.iterrows():
                emb_a = embed(model, row["img_a"], transform)
                emb_b = embed(model, row["img_b"], transform)
                dist = torch.nn.functional.pairwise_distance(emb_a, emb_b).item()
                all_dists.append(dist)
                all_labels.append(row["label"])

            all_dists = np.array(all_dists)
            all_labels = np.array(all_labels)

            best_acc, best_thresh = 0, 0
            for thresh in np.arange(0.1, 2.0, 0.02):
                preds = (all_dists < thresh).astype(int)
                acc = (preds == all_labels).mean()
                if acc > best_acc:
                    best_acc, best_thresh = acc, thresh

            preds = (all_dists < best_thresh).astype(int)
            far = ((preds == 1) & (all_labels == 0)).sum() / (all_labels == 0).sum()
            frr = ((preds == 0) & (all_labels == 1)).sum() / (all_labels == 1).sum()

            results.append({
                "fold": fold_num, "corruption": corruption_name, "severity": severity_idx,
                "accuracy": best_acc, "far": far, "frr": frr,
            })
            print(f"  Fold {fold_num} | {corruption_name} sev={severity_idx}: acc={best_acc:.4f} FAR={far:.4f} FRR={frr:.4f}")

    return results

if __name__ == "__main__":
    all_results = []
    for fold_num in range(1, 6):
        print(f"\n=== Fold {fold_num} ===")
        all_results.extend(evaluate_fold_corruption(fold_num))

    results_df = pd.DataFrame(all_results)
    print("\n=== Summary: mean accuracy by corruption type and severity ===")
    print(results_df.groupby(["corruption", "severity"])["accuracy"].mean().unstack())
    results_df.to_csv("metadata/corruption_results.csv", index=False)