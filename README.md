# FedCVR – Federated Cardiovascular Risk Prediction

Anonymous implementation accompanying the manuscript *"Overcoming Clinical
Data Heterogeneity: A Secure Federated Framework for Cardiovascular Risk
Prediction"* (under double-blind review).

---

## Overview

FedCVR is a federated learning framework for cardiovascular risk prediction
across institutionally siloed, heterogeneous clinical datasets. It combines:

**Adaptive server aggregation** — instead of plain weighted averaging, the
server applies a bias-corrected Adam-style moment estimator to the
aggregated pseudo-gradient, stabilising convergence under non-IID data.

**Client-side Differential Privacy** — each client clips its model update
to L2 norm `C` and adds calibrated Gaussian noise `N(0, σ²C²I)` before
transmission, so the server never observes an unperturbed update
(client-level DP).

```
                ┌──────────────────────────────────────────────────┐
                │                   FL Server                      │
                │   w_{t+1} = w_t + η · m̂_t / (√v̂_t + ε_opt)      │
                │   (bias-corrected adaptive moment aggregation)   │
                └────────┬──────────────┬──────────────────────────┘
                         │              │  broadcast w_t
           ┌─────────────┘              └──────────────┐
           ▼                                           ▼
  ┌──────────────────┐                       ┌──────────────────┐
  │  Client 0        │        · · ·          │  Client 4        │
  │  Framingham      │                       │  Long Beach VA   │
  │                  │                       │                  │
  │  1. Local Adam   │                       │  1. Local Adam   │
  │     (5 epochs)   │                       │     (5 epochs)   │
  │  2. DP: clip Δθ  │                       │  2. DP: clip Δθ  │
  │     + Gaussian   │                       │     + Gaussian   │
  │     noise        │                       │     noise        │
  └──────────────────┘                       └──────────────────┘
```

---

## Method summary (matches the manuscript)

### Model architecture (Section 3.2)

```
Input(13) → Linear(64) → ReLU → Dropout(0.3)
          → Linear(32) → ReLU → Dropout(0.3)
          → Linear(1)  → Sigmoid
```

Trained with binary cross-entropy (`BCELoss`) on sigmoid probabilities,
Adam optimiser (lr = 0.001), batch size 32, 5 local epochs per round,
Xavier initialisation.

### Client-side Differential Privacy (Equations 8 and 9)

```
Δθ      = θ_local − θ_global
Δθ_clip = Δθ · min(1, C / ‖Δθ‖₂)          (C = 1.0)
Δθ̃      = Δθ_clip + N(0, σ²C²I)
```

The perturbed update is transmitted; the unperturbed update never leaves
the institution (client-level differential privacy).

### Server aggregation (Equations 3 to 6)

```
Δ_t  = FedAvg(client_parameters) − w_t
m_t  = β₁·m_{t-1} + (1−β₁)·Δ_t            (β₁ = 0.9)
v_t  = β₂·v_{t-1} + (1−β₂)·Δ_t²           (β₂ = 0.999)
m̂_t  = m_t / (1 − β₁ᵗ)
v̂_t  = v_t / (1 − β₂ᵗ)
w_{t+1} = w_t + η·m̂_t / (√v̂_t + ε_opt)    (η = 1.0, ε_opt = 1e-8)
```

Setting `η = 0` disables the server optimiser (plain FedAvg baseline).

### Datasets (Section 3.3.1)

| Client | Institution | File |
|--------|-------------|------|
| 0 | Framingham Heart Study | `framingham.csv` |
| 1 | Cleveland Clinic Foundation | `cleveland.csv` |
| 2 | Hungarian Institute of Cardiology | `hungarian.csv` |
| 3 | University Hospital Zurich (Switzerland) | `switzerland.csv` |
| 4 | Long Beach VA Medical Center | `long_beach_va.csv` |

All sources are harmonised to the 13-attribute UCI Heart Disease schema.
Attributes absent from a source (notably in the Framingham study) are
treated as missing and imputed with site-specific medians computed on the
training partition only. See `data/README.md` for download instructions.

### Evaluation protocol (Section 3.3.3)

1. Stratified 80/20 train/test split per institution BEFORE federation.
2. Client-level 5-fold cross-validation (leave-one-institution-out).
3. Final global model evaluated on the composite test set aggregating the
   20% test partitions of all institutions.
4. Paired t-tests with Bonferroni correction across folds.

---

## Repository structure

```
fedcvr/
├── fedcvr/
│   ├── model.py              # 13→64→32→1 DNN, dropout 0.3, sigmoid output
│   ├── client.py             # Local Adam training + update-level DP
│   ├── strategy.py           # Adaptive (Adam-style) server aggregation
│   └── data_utils.py         # Harmonisation to 13 features, leak-free splits
│
├── experiments/
│   ├── run_cross_validation.py   # Leave-one-client-out CV + global model
│   ├── run_statistical_tests.py  # Paired t-tests, Bonferroni (Table 5)
│   ├── run_comparison.py         # Convergence curves (Figure + speeds)
│   └── run_dp_sensitivity.py     # Privacy-utility trade-off (Table 6)
│
├── data/                     # Place the five CSVs here (see data/README.md)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All experiments run on conventional CPU hardware; the compact model size
makes GPU acceleration unnecessary.

---

## Reproducing the paper's results

```bash
# 1) Cross-validation protocol + global model (Tables 1, 2, 3)
python -m experiments.run_cross_validation \
    --data_dir data --rounds 100 --seeds 42 43 44 45 46 --out_dir results

# 2) Statistical significance tests (Table 5)
python -m experiments.run_statistical_tests \
    --cv_csv results/cv_fold_results.csv \
    --out_csv results/statistical_tests.csv

# 3) Convergence analysis (Figure: learning curves)
python -m experiments.run_comparison --data_dir data --rounds 100

# 4) DP sensitivity analysis (Table 6, privacy-utility figure)
python -m experiments.run_dp_sensitivity --data_dir data --rounds 100
```

Every number reported in the manuscript is traceable to the CSV files
produced by these four scripts.

---

## License

Released under the MIT License (anonymised for double-blind review; see
`LICENSE`). The datasets are subject to their own respective licenses;
refer to `data/README.md` for the original sources.
