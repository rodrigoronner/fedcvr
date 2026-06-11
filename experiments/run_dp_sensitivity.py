"""
experiments/run_dp_sensitivity.py
==================================
Investigation 3 of the paper – Differential Privacy sensitivity analysis.

Evaluates the privacy-utility trade-off of FedCVR across the four DP
regimes of Section 3.3.4 of the paper, using update-level client-side DP
(Equations 8 and 9):

    No Privacy      (σ = 0.0, ε = ∞)
    Low Privacy     (σ = 0.8)
    Medium Privacy  (σ = 1.1)
    High Privacy    (σ = 1.5)

All regimes run for the full 100 communication rounds with batch size 32,
matching the experimental protocol of the paper. Final metrics are
computed on the composite global test set, producing the values for the
paper's privacy-impact table (Table 6) and the privacy-utility figure.

Outputs
-------
  results/dp_sensitivity_metrics.csv – round-level metrics per regime
  results/dp_final_metrics.csv       – final global test metrics (Table 6)
  results/dp_sensitivity_plot.png    – privacy-utility trade-off figure

Usage
-----
    python -m experiments.run_dp_sensitivity --data_dir data --rounds 100
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Optional

import flwr as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_cross_validation import evaluate_arrays, params_to_model
from fedcvr.client import build_client
from fedcvr.data_utils import (
    aggregate_metrics_fn,
    build_global_test_set,
    load_and_preprocess_data,
    set_global_seeds,
)
from fedcvr.strategy import FedCVRStrategy

SERVER_ETA = 1.0  # Section 3.5 of the paper

DP_SCENARIOS: Dict[str, Optional[Dict]] = {
    "No Privacy (σ=0.0)": None,
    "Low Privacy (σ=0.8)": {"noise_multiplier": 0.8, "max_grad_norm": 1.0},
    "Medium Privacy (σ=1.1)": {"noise_multiplier": 1.1, "max_grad_norm": 1.0},
    "High Privacy (σ=1.5)": {"noise_multiplier": 1.5, "max_grad_norm": 1.0},
}

LINE_STYLES = {
    "No Privacy (σ=0.0)": ("-", "tab:blue"),
    "Low Privacy (σ=0.8)": ("--", "tab:orange"),
    "Medium Privacy (σ=1.1)": (":", "tab:green"),
    "High Privacy (σ=1.5)": ("-.", "tab:red"),
}


def run(data_dir: str, num_rounds: int, out_dir: str, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    client_train, client_test, _ = load_and_preprocess_data(data_dir=data_dir)
    if client_train is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    num_clients = len(client_train)
    input_features = client_train[0][0].shape[1]
    X_glob, y_glob = build_global_test_set(client_test)

    history_storage: Dict[str, fl.server.history.History] = {}
    final_rows = []

    for name, dp_cfg in DP_SCENARIOS.items():
        print(f"\n{'='*60}\n  Running: {name}\n{'='*60}")
        set_global_seeds(seed)

        def make_client_fn(dp_config):
            def client_fn(cid: str) -> fl.client.Client:
                return build_client(
                    cid=cid,
                    client_train_data=client_train,
                    client_test_data=client_test,
                    local_epochs=5,
                    use_dp=dp_config is not None,
                    dp_config=dp_config,
                    seed=seed,
                ).to_client()
            return client_fn

        strategy = FedCVRStrategy(
            eta=SERVER_ETA,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(dp_cfg),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        history_storage[name] = history

        # Final global-test-set metrics for Table 6
        model = params_to_model(strategy.final_weights, input_features)
        sigma = dp_cfg["noise_multiplier"] if dp_cfg else 0.0
        final_rows.append(
            {"regime": name, "sigma": sigma, **evaluate_arrays(model, X_glob, y_glob)}
        )

    # Round-level CSV
    rows = []
    for name, hist in history_storage.items():
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            for rnd, val in hist.metrics_distributed.get(metric, []):
                rows.append(
                    {"scenario": name, "round": rnd, "metric": metric, "value": val}
                )
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "dp_sensitivity_metrics.csv"), index=False
    )
    pd.DataFrame(final_rows).to_csv(
        os.path.join(out_dir, "dp_final_metrics.csv"), index=False
    )

    # 2x2 plot
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics_to_plot):
        for name, hist in history_storage.items():
            data = hist.metrics_distributed.get(metric, [])
            if data:
                rounds, values = zip(*data)
                ls, col = LINE_STYLES[name]
                ax.plot(rounds, values, label=name, linestyle=ls, color=col,
                        marker=".", markersize=4, alpha=0.9)
        ax.set_title(metric.replace("_", " ").capitalize(), fontsize=13)
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
    for ax in axes[2:]:
        ax.set_xlabel("Federated Round", fontsize=11)
    fig.suptitle("FedCVR – Privacy-Utility Trade-off (DP Sensitivity)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "dp_sensitivity_plot.png"), dpi=150,
                bbox_inches="tight")
    print(f"\nOutputs written to {out_dir}/")
    print("  dp_final_metrics.csv → Table 6 (use ONLY these values)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedCVR – DP sensitivity analysis")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.data_dir, args.rounds, args.out_dir, args.seed)
