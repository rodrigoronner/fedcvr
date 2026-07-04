# FedCVR: Validating Adaptive Federated Learning Under Differential Privacy for Cardiovascular Risk Prediction

**Paper:** *Validating the FedCVR Framework on Real Heterogeneous Clinical Datasets: Adaptive Federated Learning Under Differential Privacy for Cardiovascular Risk Prediction*
**Authors:** Rodrigo Tertulino, Ricardo Almeida, Laercio Alencar
**Affiliation:** IFRN - Federal Institute of Education, Science and Technology of Rio Grande do Norte, Mossoro, RN, Brazil
**Repository:** https://github.com/rodrigoronner/fedcvr

---

## Overview

FedCVR validates the server-side adaptive aggregation mechanism introduced in Tertulino and Alencar (2026) on five real publicly available cardiovascular datasets, extending the original synthetic-data case study to genuine multi-institutional heterogeneous data. The framework addresses two simultaneous challenges in clinical federated learning: statistical heterogeneity (non-IID data) and Differential Privacy noise corruption.

**Key architectural components:**

- **Adaptive server aggregation:** Adam-style bias-corrected moment estimation acting as a temporal denoiser for DP noise (Equations 3 to 6 in the paper).
- **Client-level Differential Privacy:** update-level gradient clipping and Gaussian noise injection applied to the model update vector before transmission (Equations 8 and 9). The server never receives an unperturbed update.
- **13-feature UCI schema:** harmonized feature set covering the five real cardiovascular datasets.
- **Leave-one-institution-out cross-validation:** client-level evaluation protocol measuring generalization to entirely unseen healthcare institutions.

---

## Server Update Rule (Equation 6)

```
Delta_t  = FedAvg(client_parameters) - w_t       (pseudo-gradient)
m_t      = beta_1 * m_{t-1} + (1 - beta_1) * Delta_t
v_t      = beta_2 * v_{t-1} + (1 - beta_2) * Delta_t^2
m_hat_t  = m_t / (1 - beta_1^t)
v_hat_t  = v_t / (1 - beta_2^t)
w_{t+1}  = w_t + eta * m_hat_t / (sqrt(v_hat_t) + eps_opt)
```

Default: eta=1.0, beta_1=0.9, beta_2=0.999, eps_opt=1e-8. Setting eta=0.0 gives plain FedAvg.

## Client-Level DP (Equations 8 and 9)

```
Delta_clip  = Delta_theta * min(1, C / ||Delta_theta||_2)   (Eq. 8, clipping)
Delta_noisy = Delta_clip + N(0, sigma^2 * C^2 * I)          (Eq. 9, noise)
```

Default: C=1.0, sigma in {0.0, 0.8, 1.1, 1.5}.

## Model Architecture (Section 3.2)

```
Input(13) -> Linear(64) -> ReLU -> Dropout(0.3)
          -> Linear(32) -> ReLU -> Dropout(0.3)
          -> Linear(1)  -> Sigmoid
```

Trained with BCELoss and Adam (lr=0.001). All linear layers use Xavier uniform initialization.

---

## Datasets

| Client | Institution | File |
|--------|-------------|------|
| 0 | Framingham Heart Study | framingham.csv |
| 1 | Cleveland Clinic Foundation | cleveland.csv |
| 2 | Hungarian Institute of Cardiology | hungarian.csv |
| 3 | University Hospital Zurich (Switzerland) | switzerland.csv |
| 4 | Long Beach VA Medical Center | long_beach_va.csv |

Download instructions: see `data/README.md`.

---

## Repository Structure

```
fedcvr/
├── fedcvr/
│   ├── model.py          # 13-feature DNN: 64/32 hidden, Dropout(0.3), Sigmoid
│   ├── client.py         # Local Adam training + update-level client-level DP
│   ├── strategy.py       # FedCVRStrategy: Adam-style server aggregation
│   └── data_utils.py     # Harmonization to 13-attr UCI schema
│
├── experiments/
│   ├── run_cross_validation.py   # Leave-one-client-out CV + global model (Tables 1, 2, 3)
│   ├── run_statistical_tests.py  # Paired t-tests, Bonferroni correction (Table 5)
│   ├── run_comparison.py         # Convergence curves (Figure)
│   └── run_dp_sensitivity.py     # Privacy-utility trade-off (Table 6)
│
├── data/                         # Place CSV files here (see data/README.md)
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/rodrigoronner/fedcvr.git
cd fedcvr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Paper Results

```bash
# 1) Cross-validation + global model (Tables 1, 2, 3)
python -m experiments.run_cross_validation \
    --data_dir data --rounds 100 --seeds 42 43 44 45 46

# 2) Statistical significance tests (Table 5)
python -m experiments.run_statistical_tests \
    --cv_csv results/cv_fold_results.csv

# 3) Convergence comparison (Figure)
python -m experiments.run_comparison --data_dir data --rounds 100

# 4) DP sensitivity analysis (Table 6)
python -m experiments.run_dp_sensitivity --data_dir data --rounds 100
```

Every number in the paper is traceable to the CSV files produced by these four scripts.

---

## Hyperparameters (Section 3.5)

| Parameter | Value |
|-----------|-------|
| Local optimizer | Adam, lr=0.001 |
| Server learning rate (eta) | 1.0 |
| beta_1 | 0.9 |
| beta_2 | 0.999 |
| eps_opt | 1e-8 |
| Batch size | 32 |
| Local epochs | 5 |
| Communication rounds | 100 |
| Clipping norm C | 1.0 |
| DP sigma grid | {0.0, 0.8, 1.1, 1.5} |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tertulino2026fedcvr,
  author    = {Tertulino, Rodrigo and Almeida, Ricardo and Alencar, Laercio},
  title     = {Validating the {FedCVR} Framework on Real Heterogeneous Clinical Datasets:
               Adaptive Federated Learning Under Differential Privacy for Cardiovascular Risk Prediction},
  journal   = {Journal of the Brazilian Computer Society},
  year      = {2026},
  publisher = {Springer}
}
```

## License

MIT License. See `LICENSE` for details. The datasets are subject to their own respective licenses; see `data/README.md`.
