# Signature Forgery Detection

Writer-independent offline signature verification, stress-tested for realistic bank
deployment conditions: low enrollment sample counts, degraded image capture, and
generalization to scripts never seen in training. Two models are compared on identical
folds, pairs and seeds: a Siamese CNN on grayscale pixels and a Siamese GNN on a graph of
stroke points (EdgeConv), plus a binarized-CNN control.

An earlier classical-ML baseline (HOG features + Logistic Regression/KNN/SVM/Random Forest)
reported 98.9% accuracy on CEDAR, but used a writer-dependent evaluation split, inflating the
result. This project rebuilds the pipeline around a writer-disjoint evaluation and a
Siamese/metric-learning architecture that actually generalizes to unseen writers.

## Results summary

Same writer-disjoint folds (5 folds, 45 train / 10 test writers), same pairs, same seeds.
Thresholds are tuned on the evaluated pairs, so every accuracy is a best-case figure.

| Experiment | Siamese CNN | Siamese GNN |
|---|---|---|
| CEDAR, writer-disjoint, 5-fold | 99.95% | 67.65% |
| Binarized-CNN control (same CNN on Otsu-binarized input) | 75.00% (20 ep), 75.35% (60 ep) | n/a |
| Few-shot enrollment (n = 3, 5, 10) | no degradation | 70.1%, 70.9%, 72.4% |
| JPEG quality 5 | 68% | 61% |
| Brightness shift (0.1 to 0.5) | about 51% to 56% (near chance) | about 65% to 68% |
| Perspective warp 0.16 | 98% | 57.5% |
| Cross-script, Bengali | 71.5% | 60.0% |
| Cross-script, Hindi | 74.3% | 60.3% |
| Late fusion of CNN and GNN distances | no gain over the CNN alone | |

Main reading: the CNN is far more accurate, but it relies on gray-level cues that break under
brightness shift. The GNN is less accurate and is insensitive to brightness because it only
sees binarized stroke geometry, but it is weaker under perspective warp. The binarized-CNN
control (75%) shows that most of the GNN's gap comes from losing gray-level information, not
from the graph design alone.

An earlier classical baseline (HOG + Logistic Regression/KNN/SVM/Random Forest) reported
98.9% on CEDAR using a writer-dependent split, which inflates the result. It was not re-run
under a writer-disjoint split.

Full methodology, per-fold numbers, and the two brightness-fix attempts (both documented as
negative results) are in the project report. Figures are in `figures/`.

## Datasets

- **CEDAR** — 55 writers, 24 genuine + 24 forged signatures each, Latin script. Used for
  training and the primary baseline.
- **BHSig260** — 100 Bengali + 160 Hindi writers, 24 genuine + 30 forged each. Used **only**
  for the cross-script generalization test — never seen during training.

Datasets are not included in this repo. Download separately:
- CEDAR: search Kaggle for "CEDAR signature dataset"
- BHSig260: search Kaggle for "BHSig260"

Place them under `data/` — see folder structure expected by `src/build_metadata.py` and
`src/buildbhsigmetadata.py`.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

GPU (CUDA) strongly recommended for training. Verify PyTorch sees your GPU before running
anything:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Project structure

```
forgery/
├── data/              # raw CEDAR / BHSig260 images (not included, download separately)
├── metadata/          # generated CSVs, fold splits, trained model weights
├── src/                # all pipeline code
│   ├── build_metadata.py       # parses CEDAR filenames -> metadata table
│   ├── buildbhsigmetadata.py   # parses BHSig260 filenames -> metadata table
│   ├── split.py                # writer-disjoint train/test split (reference implementation)
│   ├── bpairs.py                # builds genuine/forged image pairs for training
│   ├── dataset.py              # PyTorch Dataset — loads and preprocesses image pairs
│   ├── model.py                # SiameseCNN architecture
│   ├── train.py                # contrastive loss + single-run training loop
│   ├── multifold.py            # main experiment driver — 5-fold CV, trains + saves all fold models
│   ├── evaluate.py             # threshold sweep + accuracy/FAR/FRR on a trained model
│   ├── fewshotseval.py         # few-shot enrollment robustness sweep
│   ├── corrupteval.py          # image corruption robustness sweep
│   ├── crossscript_eval.py     # CEDAR -> BHSig260 cross-script transfer test
│   ├── graphdata.py            # image -> graph (Otsu, farthest-point sampling, kNN) and graph pair dataset
│   ├── gnn_model.py            # SiameseGNN (EdgeConv, mean+max readout)
│   ├── multifold_gnn.py        # 5-fold GNN training and evaluation
│   ├── multifold_bin.py        # binarized-CNN control, 5 folds
│   ├── gnn_eval_all.py         # GNN cross-script, few-shot and corruption evaluation
│   ├── fusion.py               # late fusion of CNN and GNN distances
│   └── make_figures.py         # builds all report figures into figures/
├── figures/            # report figures (PNG)
├── debug/              # diagnostic scripts used during development (kept as record)
├── viz/                # architecture and attribution (occlusion, SHAP) exports
└── requirements.txt
```

## Running the pipeline

All commands run from the project root.

```bash
# 1. Build metadata tables from raw images
python src/build_metadata.py
python src/buildbhsigmetadata.py

# 2. Train + evaluate baseline (5-fold cross-validation)
python src/multifold.py

# 3. Run robustness experiments (uses the fold models saved by step 2)
python src/fewshotseval.py
python src/corrupteval.py
python src/crossscript_eval.py

# 4. GNN and binarized-CNN control
python src/multifold_gnn.py
python src/multifold_bin.py
python src/gnn_eval_all.py --what all      # or crossscript | fewshot | corruption

# 5. Fusion and figures
python src/fusion.py
python src/make_figures.py
```

Each script prints per-fold results and saves a summary CSV under `metadata/`.

## Key findings

- **Writer-dependent evaluation inflates accuracy.** Always split by writer ID, not randomly,
  for signature verification — see `src/split.py`'s leakage assertion.
- **The model is highly sensitive to brightness shift**, even mild ones, despite being robust
  to blur and perspective distortion. Two remediation attempts (input normalization,
  training-time brightness augmentation) were tried; both improved brightness robustness at
  the cost of degrading performance elsewhere. Neither is used in the final reported model —
  this is documented as an open limitation.
- **The CNN and GNN fail in different places.** The CNN is near chance under brightness shift
  but robust to blur and perspective. The GNN is the reverse on brightness and perspective,
  at a much lower clean accuracy. Late fusion did not beat the CNN alone.
- **Most of the GNN gap is information loss, not architecture.** The same CNN on binarized
  input drops from 99.95% to about 75%, so the grayscale cues carry much of the CNN's accuracy.
  Whether those cues are real handwriting properties or CEDAR scan artifacts is untested.
- **Cross-script generalization is real but limited.** A model trained only on Latin-script
  signatures (CEDAR) drops from ~100% to ~71–74% on unseen Bengali/Hindi signatures
  (BHSig260), and the CNN's failure skews toward false acceptance of forgeries, the more
  dangerous error type for a real deployment. The GNN reaches about 60% on both scripts.

## Limitations

- Thresholds are tuned on the evaluated pairs (no separate validation split), so all
  accuracies are best-case.
- Corruptions are synthetic and were not validated against real phone photos.
- The cause of the brightness failure is not isolated (absolute-intensity reliance,
  independent per-image shifts, or background clipping are all candidates).
- The GNN node features are coordinates only, and the GNN was not tuned beyond one design.
- Experimental v2 GNN scripts, if present in the repo, are unfinished and are not used in any
  reported number.

## License

Add your license here.## Running the pipeline

All commands run from the project root.

```bash
# 1. Build metadata tables from raw images
python src/build_metadata.py
python src/buildbhsigmetadata.py

# 2. Train + evaluate baseline (5-fold cross-validation)
python src/multifold.py

# 3. Run robustness experiments (uses the fold models saved by step 2)
python src/fewshotseval.py
python src/corrupteval.py
python src/crossscript_eval.py
```

Each script prints per-fold results and saves a summary CSV under `metadata/`.

## Key findings

- **Writer-dependent evaluation inflates accuracy.** Always split by writer ID, not randomly,
  for signature verification — see `src/split.py`'s leakage assertion.
- **The model is highly sensitive to brightness shift**, even mild ones, despite being robust
  to blur and perspective distortion. Two remediation attempts (input normalization,
  training-time brightness augmentation) were tried; both improved brightness robustness at
  the cost of degrading performance elsewhere. Neither is used in the final reported model —
  this is documented as an open limitation.
- **Cross-script generalization is real but limited.** A model trained only on Latin-script
  signatures (CEDAR) drops from ~100% to ~71–74% on unseen Bengali/Hindi signatures
  (BHSig260), and the failure specifically skews toward false acceptance of forgeries — the
  more dangerous error type for a real deployment.

## License

Add your license here.
