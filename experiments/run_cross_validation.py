"""
experiments/run_cross_validation.py
====================================
Implements the evaluation protocol of Section 3.3.3 of the paper:

1. CLIENT-LEVEL 5-FOLD CROSS-VALIDATION (leave-one-institution-out):
   in each fold, one complete client institution is held out entirely;
   the federated model is trained on the remaining four clients and then
   evaluated on the held-out client's test partition. This measures
   generalisation to an entirely unseen healthcare institution.

2. GLOBAL MODEL + COMPOSITE TEST SET: a final model is trained on all
   five clients and evaluated on the aggregation of the 20% test
   partitions of every institution (never exposed to training).
   Produces the numbers for the paper's baseline table (Table 1) and the
   per-client breakdown (Table 2).

Each configuration can be repeated with multiple seeds (--seeds) so the
fold-level results can feed the paired statistical tests
(experiments/run_statistical_tests.py).

Outputs (in --out_dir)
----------------------
  cv_fold_results.csv      – one row per (strategy, seed, held-out client)
  global_model_metrics.csv – global model on composite test set, per seed
  per_client_metrics.csv   – global model evaluated per institution (Table 2)

Usage
-----
    python -m experiments.run_cross_validation --data_dir data \
        --rounds 100 --seeds 42 43 44 45 46 --out_dir results
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcvr.client import build_client
from fedcvr.data_utils import (
    CLIENT_NAMES,
    aggregate_metrics_fn,
    build_global_test_set,
    load_and_preprocess_data,
    set_global_seeds,
)
from fedcvr.model import Net
from fedcvr.strategy import FedCVRStrategy


# ---------------------------------------------------------------------------
# Strategy configurations evaluated in the paper (Table 3)
# ---------------------------------------------------------------------------
# FedAvg          : plain weighted averaging (eta = 0 disables server Adam)
# FedCVR-NoDP     : adaptive server aggregation, no Differential Privacy
# FedCVR-Complete : adaptive aggregation + client-side DP (sigma = 0.8)

STRATEGIES: Dict[str, Dict] = {
    "FedAvg": {"eta": 0.0, "dp": None},
    "FedCVR-NoDP": {"eta": 1.0, "dp": None},
    "FedCVR-Complete": {"eta": 1.0, "dp": {"noise_multiplier": 0.8, "max_grad_norm": 1.0}},
}


def evaluate_arrays(model: Net, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a model on numpy arrays and return the standard metrics."""
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1_score": float(f1_score(y, preds, zero_division=0)),
        "auc": auc,
    }


def params_to_model(parameters: fl.common.Parameters, input_features: int) -> Net:
    """Materialise a Net from Flower Parameters."""
    from collections import OrderedDict

    nds = fl.common.parameters_to_ndarrays(parameters)
    model = Net(input_features=input_features)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), nds)}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def run_federation(
    train_subset: List[Tuple[np.ndarray, np.ndarray]],
    test_subset: List[Tuple[np.ndarray, np.ndarray]],
    strategy_cfg: Dict,
    num_rounds: int,
    seed: int,
) -> fl.common.Parameters:
    """Run one federated training and return the final global parameters."""
    set_global_seeds(seed)
    num_clients = len(train_subset)

    def client_fn(cid: str) -> fl.client.Client:
        return build_client(
            cid=cid,
            client_train_data=train_subset,
            client_test_data=test_subset,
            local_epochs=5,
            use_dp=strategy_cfg["dp"] is not None,
            dp_config=strategy_cfg["dp"],
            seed=seed,
        ).to_client()

    strategy = FedCVRStrategy(
        eta=strategy_cfg["eta"],
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    return strategy.final_weights


def run(data_dir: str, num_rounds: int, seeds: List[int], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    client_train, client_test, names = load_and_preprocess_data(data_dir=data_dir)
    if client_train is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    input_features = client_train[0][0].shape[1]
    fold_rows, global_rows, per_client_rows = [], [], []

    for seed in seeds:
        for strat_name, strat_cfg in STRATEGIES.items():

            # ---------------------------------------------------------
            # 1) Leave-one-client-out cross-validation (Section 3.3.3)
            # ---------------------------------------------------------
            for held_out in range(len(names)):
                print(
                    f"\n=== seed={seed} | {strat_name} | fold: hold out "
                    f"{names[held_out]} ==="
                )
                train_subset = [
                    d for i, d in enumerate(client_train) if i != held_out
                ]
                test_subset = [
                    d for i, d in enumerate(client_test) if i != held_out
                ]
                final_params = run_federation(
                    train_subset, test_subset, strat_cfg, num_rounds, seed
                )
                model = params_to_model(final_params, input_features)
                X_ho, y_ho = client_test[held_out]
                metrics = evaluate_arrays(model, X_ho, y_ho)
                fold_rows.append(
                    {
                        "strategy": strat_name,
                        "seed": seed,
                        "held_out_client": names[held_out],
                        **metrics,
                    }
                )

            # ---------------------------------------------------------
            # 2) Global model on all five clients + composite test set
            # ---------------------------------------------------------
            print(f"\n=== seed={seed} | {strat_name} | global model ===")
            final_params = run_federation(
                client_train, client_test, strat_cfg, num_rounds, seed
            )
            model = params_to_model(final_params, input_features)

            X_glob, y_glob = build_global_test_set(client_test)
            global_rows.append(
                {
                    "strategy": strat_name,
                    "seed": seed,
                    **evaluate_arrays(model, X_glob, y_glob),
                }
            )

            for i, (X_te, y_te) in enumerate(client_test):
                per_client_rows.append(
                    {
                        "strategy": strat_name,
                        "seed": seed,
                        "client": names[i],
                        "n_test": len(y_te),
                        **evaluate_arrays(model, X_te, y_te),
                    }
                )

    pd.DataFrame(fold_rows).to_csv(
        os.path.join(out_dir, "cv_fold_results.csv"), index=False
    )
    pd.DataFrame(global_rows).to_csv(
        os.path.join(out_dir, "global_model_metrics.csv"), index=False
    )
    pd.DataFrame(per_client_rows).to_csv(
        os.path.join(out_dir, "per_client_metrics.csv"), index=False
    )
    print(f"\nResults written to {out_dir}/")
    print("  cv_fold_results.csv      → Table 3 (means and std across folds)")
    print("  global_model_metrics.csv → Table 1")
    print("  per_client_metrics.csv   → Table 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FedCVR – client-level cross-validation protocol"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    run(args.data_dir, args.rounds, args.seeds, args.out_dir)
