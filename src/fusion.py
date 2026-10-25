"""Late fusion of the grayscale CNN and the ORIGINAL GNN (gnn_model.pth in each fold folder).
Run from the project root:  python src\\fusion_eval_v1.py
Reports AUC (threshold-free) and best-threshold accuracy for: CNN alone, GNN alone, equal-weight fusion, best-weight fusion."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def zs(d):
    return (d - d.mean()) / (d.std() + 1e-9)       # standardised within one evaluation set (distances only, no labels)


def best_acc(d, y):
    ts = np.quantile(d, np.linspace(0.01, 0.99, 197))
    return max(((d < t).astype(int) == y).mean() for t in ts)


def score(d, y):
    return roc_auc_score(y, -d), best_acc(d, y)    # label 1 = same writer = small distance


def fuse_eval(d_cnn, d_gnn, y, weights=np.linspace(0, 1, 11)):
    out = {"cnn": score(d_cnn, y), "gnn": score(d_gnn, y)}
    zc, zg = zs(d_cnn), zs(d_gnn)
    out["fused_w0.5"] = score(0.5 * zc + 0.5 * zg, y)
    per_w = {round(w, 1): score(w * zc + (1 - w) * zg, y) for w in weights}
    wbest = max(per_w, key=lambda w: per_w[w][0])
    out["fused_bestw"] = per_w[wbest]
    out["_best_w"] = wbest
    return out


def main():
    import torch
    from torch.utils.data import DataLoader
    from dataset import SignaturePairDataset
    from model import SiameseCNN
    from graphdata import GraphPairDataset
    from gnn_model import SiameseGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    def dists_cnn(model, csv):
        d, y = [], []
        with torch.no_grad():
            for a, b, l in DataLoader(SignaturePairDataset(csv), batch_size=64):
                ea, eb = model(a.to(device), b.to(device))
                d.extend(torch.nn.functional.pairwise_distance(ea, eb).cpu().numpy()); y.extend(l.numpy())
        return np.array(d), np.array(y)

    def dists_gnn(model, ds):
        d, y = [], []
        with torch.no_grad():
            for ca, ia, cb, ib, l in DataLoader(ds, batch_size=64):
                ea, eb = model(ca.to(device), ia.to(device), cb.to(device), ib.to(device))
                d.extend(torch.nn.functional.pairwise_distance(ea, eb).cpu().numpy()); y.extend(l.numpy())
        return np.array(d), np.array(y)

    sets = {"CEDAR test": None, "BHSig260 Bengali": "metadata/bengali_pairs.csv", "BHSig260 Hindi": "metadata/hindi_pairs.csv"}
    graph_ds = {}      # graphs are cached inside each dataset, so BHSig260 graphs are built once and reused by later folds
    rows = []
    for fold in range(1, 6):
        fd = f"metadata/folds/fold_{fold}"
        cnn = SiameseCNN(embedding_dim=128).to(device); cnn.load_state_dict(torch.load(f"{fd}/siamese_model.pth")); cnn.eval()
        gnn = SiameseGNN(embedding_dim=128).to(device); gnn.load_state_dict(torch.load(f"{fd}/gnn_model.pth")); gnn.eval()
        for name, csv in sets.items():
            t0 = time.time()
            csv = csv or f"{fd}/test_pairs.csv"
            dc, y = dists_cnn(cnn, csv)
            if csv not in graph_ds:
                graph_ds[csv] = GraphPairDataset(csv)
            dg, y2 = dists_gnn(gnn, graph_ds[csv])
            assert (y == y2).all(), "pair order mismatch"
            res = fuse_eval(dc, dg, y)
            res.pop("_best_w")
            for method, (auc, acc) in res.items():
                rows.append({"set": name, "fold": fold, "method": method, "auc": auc, "acc": acc})
            print(f"fold {fold} | {name}: {len(y)} pairs, {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print(df.groupby(["set", "method"])[["auc", "acc"]].mean().round(4))
    df.to_csv("metadata/fusion_v1_results.csv", index=False)


if __name__ == "__main__":
    main()
