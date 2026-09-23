# Signature Forgery Detection

Writer-independent offline signature verification using a Siamese CNN, stress-tested for
realistic bank deployment conditions: low enrollment sample counts, degraded image capture,
and generalization to scripts never seen in training.

An earlier classical-ML baseline (HOG features + Logistic Regression/KNN/SVM/Random Forest)
reported 98.9% accuracy on CEDAR, but used a writer-dependent evaluation split, inflating the
result. This project rebuilds the pipeline around a writer-disjoint evaluation and a
Siamese/metric-learning architecture that actually generalizes to unseen writers.

## Results summary

| Experiment | Result |
|---|---|
| Baseline (CEDAR, writer-disjoint, 5-fold) | 99.95% mean accuracy, FAR 0.0%, FRR 0.10% |
| Few-shot enrollment (n=3–5 references) | No measurable degradation vs. full enrollment |
| Corruption robustness | Robust to blur/perspective; sharp failure under severe JPEG compression and any tested brightness shift |
| Cross-script transfer (CEDAR → BHSig260) | 99.95% → ~71–74%, FAR-dominant failure mode |

Full methodology, per-fold numbers, and the two brightness-fix attempts (both documented as
partial/negative results) are in the project report PDF.

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
│   └── crossscript_eval.py     # CEDAR -> BHSig260 cross-script transfer test
├── debug/              # diagnostic scripts used during development (kept as record)
├── viz/                 # architecture visualization exports
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
