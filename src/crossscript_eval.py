import torch
import numpy as np
import pandas as pd
import itertools
import random
from model import SiameseCNN
from dataset import SignaturePairDataset
from torch.utils.data import DataLoader

random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_bhsig_pairs(df, pairs_per_writer=40):
    pairs = []
    for w in df["writer_id"].unique():
        genuine = df[(df.writer_id == w) & (df.label == "genuine")]["filepath"].tolist()
        forged = df[(df.writer_id == w) & (df.label == "forged")]["filepath"].tolist()

        genuine_combos = list(itertools.combinations(genuine, 2))
        random.shuffle(genuine_combos)
        for a, b in genuine_combos[:pairs_per_writer // 2]:
            pairs.append({"img_a": a, "img_b": b, "label": 1, "writer_id": w})

        gf_combos = list(itertools.product(genuine, forged))
        random.shuffle(gf_combos)
        for a, b in gf_combos[:pairs_per_writer // 2]:
            pairs.append({"img_a": a, "img_b": b, "label": 0, "writer_id": w})

    pairs_df = pd.DataFrame(pairs)
    return pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

def evaluate_on_script(model, pairs_csv_path):
    ds = SignaturePairDataset(pairs_csv_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    all_dists, all_labels = [], []
    with torch.no_grad():
        for img_a, img_b, label in loader:
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
    return best_acc, best_thresh, far, frr

if __name__ == "__main__":
    bhsig_df = pd.read_csv("metadata/bhsig260_metadata.csv")
    bengali_df = bhsig_df[bhsig_df.script == "bengali"]
    hindi_df = bhsig_df[bhsig_df.script == "hindi"]

    bengali_pairs = build_bhsig_pairs(bengali_df)
    hindi_pairs = build_bhsig_pairs(hindi_df)
    bengali_pairs.to_csv("metadata/bengali_pairs.csv", index=False)
    hindi_pairs.to_csv("metadata/hindi_pairs.csv", index=False)

    print(f"Bengali pairs: {len(bengali_pairs)}, Hindi pairs: {len(hindi_pairs)}")

    results = []
    for fold_num in range(1, 6):
        model = SiameseCNN(embedding_dim=128).to(device)
        model.load_state_dict(torch.load(f"metadata/folds/fold_{fold_num}/siamese_model.pth"))
        model.eval()

        for script_name, path in [("bengali", "metadata/bengali_pairs.csv"), ("hindi", "metadata/hindi_pairs.csv")]:
            acc, thresh, far, frr = evaluate_on_script(model, path)
            print(f"Fold {fold_num} | {script_name}: acc={acc:.4f} thresh={thresh:.2f} FAR={far:.4f} FRR={frr:.4f}")
            results.append({"fold": fold_num, "script": script_name, "accuracy": acc, "threshold": thresh, "far": far, "frr": frr})

    results_df = pd.DataFrame(results)
    print("\n=== Summary: mean by script ===")
    print(results_df.groupby("script")[["accuracy", "far", "frr"]].mean())
    results_df.to_csv("metadata/crossscript_results.csv", index=False)