import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from model import SiameseCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.Grayscale(1), T.Resize((105,105)), T.ToTensor()])

def load_img(path):
    return transform(Image.open(path)).unsqueeze(0).to(device)

def embed(model, path):
    with torch.no_grad():
        return model.forward_once(load_img(path))

def evaluate_fold_fewshot(fold_num, n_values=[3, 5, 10], trials=5):
    model = SiameseCNN(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(f"metadata/folds/fold_{fold_num}/siamese_model.pth"))
    model.eval()

    test_pairs = pd.read_csv(f"metadata/folds/fold_{fold_num}/test_pairs.csv")
    test_writers = test_pairs["writer_id"].unique()

    full_meta = pd.read_csv("metadata/cedar_metadata.csv")
    fold_results = []

    for n in n_values:
        trial_accs, trial_fars, trial_frrs = [], [], []

        for trial in range(trials):
            rng = np.random.RandomState(trial)  # different reference subset each trial
            all_dists, all_labels = [], []

            for w in test_writers:
                genuine = full_meta[(full_meta.writer_id == w) & (full_meta.label == "genuine")]["filepath"].tolist()
                forged = full_meta[(full_meta.writer_id == w) & (full_meta.label == "forged")]["filepath"].tolist()

                ref_idx = rng.choice(len(genuine), size=n, replace=False)
                references = [genuine[i] for i in ref_idx]
                remaining_genuine = [g for i, g in enumerate(genuine) if i not in ref_idx]

                ref_embeddings = [embed(model, r) for r in references]

                # genuine queries (label=1, should verify as match)
                for q in remaining_genuine:
                    q_emb = embed(model, q)
                    dists = [torch.nn.functional.pairwise_distance(q_emb, r).item() for r in ref_embeddings]
                    all_dists.append(min(dists))
                    all_labels.append(1)

                # forged queries (label=0, should verify as mismatch)
                for q in forged:
                    q_emb = embed(model, q)
                    dists = [torch.nn.functional.pairwise_distance(q_emb, r).item() for r in ref_embeddings]
                    all_dists.append(min(dists))
                    all_labels.append(0)

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

            trial_accs.append(best_acc)
            trial_fars.append(far)
            trial_frrs.append(frr)

        fold_results.append({
            "fold": fold_num, "n_references": n,
            "acc_mean": np.mean(trial_accs), "acc_std": np.std(trial_accs),
            "far_mean": np.mean(trial_fars), "frr_mean": np.mean(trial_frrs),
        })
        print(f"  Fold {fold_num}, n={n}: acc={np.mean(trial_accs):.4f}±{np.std(trial_accs):.4f}, "
              f"FAR={np.mean(trial_fars):.4f}, FRR={np.mean(trial_frrs):.4f}")

    return fold_results

if __name__ == "__main__":
    all_results = []
    for fold_num in range(1, 6):
        print(f"\n=== Fold {fold_num} ===")
        all_results.extend(evaluate_fold_fewshot(fold_num))

    results_df = pd.DataFrame(all_results)
    print("\n=== Summary across folds, grouped by n_references ===")
    print(results_df.groupby("n_references")[["acc_mean", "far_mean", "frr_mean"]].mean())
    results_df.to_csv("metadata/fewshot_results.csv", index=False)