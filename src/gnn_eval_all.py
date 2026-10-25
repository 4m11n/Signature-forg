"""Run the ORIGINAL GNN (metadata/folds/fold_k/gnn_model.pth) through the same tests the CNN got.
Run from the project root (needs graphdata.py and gnn_model.py in src):
  python src\\gnn_eval_all.py --what crossscript     # BHSig260, per fold      -> metadata/gnn_crossscript_results.csv
  python src\\gnn_eval_all.py --what fewshot         # CEDAR, n = 3, 5, 10     -> metadata/gnn_fewshot_results.csv
  python src\\gnn_eval_all.py --what corruption      # 4 corruptions x 5 sev   -> metadata/gnn_corruption_results.csv
  python src\\gnn_eval_all.py --what all
Same protocol as the CNN scripts: corruption is applied to the 105x105 grayscale image before the model sees it,
and the distance threshold is the accuracy-maximising one on the evaluated pairs (so these are best-case numbers).
"""
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A

from graphdata import image_to_graph          # 105x105 -> Otsu -> 128 points -> kNN graph (same as training)

IMG = 105
CORRUPTIONS = {
    "clean": [A.NoOp()],
    "compression": [A.ImageCompression(quality_range=(q, q), p=1.0) for q in [70, 50, 30, 15, 5]],
    "blur": [A.GaussianBlur(blur_limit=(k, k), p=1.0) for k in [3, 5, 9, 13, 17]],
    "brightness": [A.RandomBrightnessContrast(brightness_limit=b, contrast_limit=0, p=1.0) for b in [0.1, 0.2, 0.3, 0.4, 0.5]],
    "perspective": [A.Perspective(scale=(s, s), p=1.0) for s in [0.02, 0.05, 0.08, 0.12, 0.16]],
}

_base, _graphs = {}, {}


def base_img(path):
    if path not in _base:
        _base[path] = np.array(Image.open(path).convert("L").resize((IMG, IMG)))
    return _base[path]


def clean_graph(path):
    if path not in _graphs:
        _graphs[path] = image_to_graph(base_img(path))
    return _graphs[path]


def make_tf(aug, seed):
    try:
        return A.Compose([aug], seed=seed)
    except TypeError:                      # older albumentations has no seed argument
        return A.Compose([aug])


def sweep(d, y):
    best_acc, best_t = 0.0, 0.0
    for t in np.arange(0.1, 2.0, 0.01):
        acc = ((d < t).astype(int) == y).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    p = (d < best_t).astype(int)
    far = ((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
    frr = ((p == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)
    return best_acc, best_t, far, frr


def embed_graphs(model, graphs, device, bs=128):
    import torch
    out = []
    for i in range(0, len(graphs), bs):
        g = graphs[i:i + bs]
        c = torch.from_numpy(np.stack([x[0] for x in g])).to(device)
        ix = torch.from_numpy(np.stack([x[1] for x in g])).to(device)
        with torch.no_grad():
            out.append(model.forward_once(c, ix).cpu().numpy())
    return np.concatenate(out)


def load_gnn(fold, device):
    import torch
    from gnn_model import SiameseGNN
    m = SiameseGNN(embedding_dim=128).to(device)
    m.load_state_dict(torch.load(f"metadata/folds/fold_{fold}/gnn_model.pth", map_location=device))
    m.eval()
    return m


def run_crossscript(folds, device):
    rows = []
    for script, csv in (("bengali", "metadata/bengali_pairs.csv"), ("hindi", "metadata/hindi_pairs.csv")):
        df = pd.read_csv(csv)
        y = df["label"].values
        t0 = time.time()
        ga = [clean_graph(p) for p in df["img_a"]]
        gb = [clean_graph(p) for p in df["img_b"]]
        print(f"{script}: built {len(set(df['img_a']) | set(df['img_b']))} graphs in {time.time()-t0:.0f}s", flush=True)
        for fold in folds:
            m = load_gnn(fold, device)
            d = np.linalg.norm(embed_graphs(m, ga, device) - embed_graphs(m, gb, device), axis=1)
            acc, t, far, frr = sweep(d, y)
            rows.append(dict(fold=fold, script=script, accuracy=acc, threshold=t, far=far, frr=frr))
            print(f"  fold {fold} | {script}: acc={acc:.4f} thr={t:.2f} FAR={far:.4f} FRR={frr:.4f}", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("script")[["accuracy", "far", "frr"]].mean().round(4))
    df.to_csv("metadata/gnn_crossscript_results.csv", index=False)


def run_fewshot(folds, device, ns=(3, 5, 10), trials=5):
    meta = pd.read_csv("metadata/cedar_metadata.csv")
    rows = []
    for fold in folds:
        m = load_gnn(fold, device)
        writers = pd.read_csv(f"metadata/folds/fold_{fold}/test_pairs.csv")["writer_id"].unique()
        sub = meta[meta.writer_id.isin(writers)]
        paths = sub["filepath"].tolist()
        emb = dict(zip(paths, embed_graphs(m, [clean_graph(p) for p in paths], device)))
        for n in ns:
            accs, fars, frrs = [], [], []
            for trial in range(trials):
                rng = np.random.RandomState(trial)
                d, y = [], []
                for w in writers:
                    gen = sub[(sub.writer_id == w) & (sub.label == "genuine")]["filepath"].tolist()
                    forg = sub[(sub.writer_id == w) & (sub.label == "forged")]["filepath"].tolist()
                    ref_idx = set(rng.choice(len(gen), size=n, replace=False).tolist())
                    refs = np.stack([emb[gen[i]] for i in sorted(ref_idx)])
                    queries = [(gen[i], 1) for i in range(len(gen)) if i not in ref_idx] + [(f, 0) for f in forg]
                    Q = np.stack([emb[q] for q, _ in queries])
                    d.extend(np.linalg.norm(Q[:, None, :] - refs[None, :, :], axis=2).min(1))
                    y.extend(l for _, l in queries)
                acc, _, far, frr = sweep(np.array(d), np.array(y))
                accs.append(acc); fars.append(far); frrs.append(frr)
            rows.append(dict(fold=fold, n_references=n, acc_mean=np.mean(accs), acc_std=np.std(accs),
                             far_mean=np.mean(fars), frr_mean=np.mean(frrs)))
            print(f"  fold {fold}, n={n}: acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}  FAR={np.mean(fars):.4f}  FRR={np.mean(frrs):.4f}", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("n_references")[["acc_mean", "far_mean", "frr_mean"]].mean().round(4))
    df.to_csv("metadata/gnn_fewshot_results.csv", index=False)


def run_corruption(folds, device):
    rows = []
    for fold in folds:
        m = load_gnn(fold, device)
        pairs = pd.read_csv(f"metadata/folds/fold_{fold}/test_pairs.csv")
        y = pairs["label"].values
        pa, pb = pairs["img_a"].tolist(), pairs["img_b"].tolist()
        for cname, augs in CORRUPTIONS.items():
            for sev, aug in enumerate(augs):
                t0 = time.time()
                tf = make_tf(aug, seed=1000 * fold + sev)       # the transform is called per image, as in the CNN sweep
                ga = [image_to_graph(tf(image=base_img(p))["image"]) for p in pa]
                gb = [image_to_graph(tf(image=base_img(p))["image"]) for p in pb]
                d = np.linalg.norm(embed_graphs(m, ga, device) - embed_graphs(m, gb, device), axis=1)
                acc, t, far, frr = sweep(d, y)
                rows.append(dict(fold=fold, corruption=cname, severity=sev, accuracy=acc, threshold=t, far=far, frr=frr))
                print(f"  Fold {fold} | {cname} sev={sev}: acc={acc:.4f} FAR={far:.4f} FRR={frr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("metadata/gnn_corruption_results.csv", index=False)     # saved after every fold
    df = pd.DataFrame(rows)
    print("\n=== Summary: mean accuracy by corruption type and severity ===")
    print(df.groupby(["corruption", "severity"])["accuracy"].mean().unstack().round(4))


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--what", default="all", choices=["all", "crossscript", "fewshot", "corruption"])
    p.add_argument("--folds", default="1,2,3,4,5")
    args = p.parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if args.what in ("all", "crossscript"):
        print("\n--- cross-script ---", flush=True); run_crossscript(folds, device)
    if args.what in ("all", "fewshot"):
        print("\n--- few-shot ---", flush=True); run_fewshot(folds, device)
    if args.what in ("all", "corruption"):
        print("\n--- corruption ---", flush=True); run_corruption(folds, device)


if __name__ == "__main__":
    main()
