"""
experiments/run_comparison.py
=============================
Investigation 2 of the paper – convergence behaviour analysis.

Trains FedAvg, FedCVR-NoDP, and FedCVR-Complete on all five clients for
100 communication rounds and records round-level metrics, producing the
learning curves (Figure: convergence analysis) and the convergence-speed
numbers reported in the paper (rounds to reach 95% of final performance).

Outputs
-------
  results/comparison_metrics.csv – round-level metrics for all strategies
  results/comparison_plot.png    – 2x2 metric comparison figure
  results/convergence_speed.csv  – rounds to reach 95% of final F1

Usage
-----
    python -m experiments.run_comparison --data_dir data --rounds 100
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict

import flwr as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcvr.client import build_client
from fedcvr.data_utils import (
    aggregate_metrics_fn,
    load_and_preprocess_data,
    set_global_seeds,
)
from fedcvr.strategy import FedCVRStrategy

SCENARIOS: Dict[str, Dict] = {
    "FedAvg": {
        "eta": 0.0, "dp": None, "linestyle": "--", "color": "tab:blue",
    },
    "FedCVR-NoDP": {
        "eta": 1.0, "dp": None, "linestyle": "-", "color": "tab:green",
    },
    "FedCVR-Complete (σ=0.8)": {
        "eta": 1.0,
        "dp": {"noise_multiplier": 0.8, "max_grad_norm": 1.0},
        "linestyle": ":",
        "color": "tab:red",
    },
}


def run(data_dir: str, num_rounds: int, out_dir: str, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    client_train, client_test, _ = load_and_preprocess_data(data_dir=data_dir)
    if client_train is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    num_clients = len(client_train)
    history_storage: Dict[str, fl.server.history.History] = {}

    for name, cfg in SCENARIOS.items():
        print(f"\n{'='*60}\n  Running: {name}\n{'='*60}")
        set_global_seeds(seed)

        def make_client_fn(dp_cfg):
            def client_fn(cid: str) -> fl.client.Client:
                return build_client(
                    cid=cid,
                    client_train_data=client_train,
                    client_test_data=client_test,
                    local_epochs=5,
                    use_dp=dp_cfg is not None,
                    dp_config=dp_cfg,
                    seed=seed,
                ).to_client()
            return client_fn

        strategy = FedCVRStrategy(
            eta=cfg["eta"],
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(cfg["dp"]),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        history_storage[name] = history

    # -------------------------------------------------------------------
    # Round-level metrics → CSV
    # -------------------------------------------------------------------
    rows = []
    for name, hist in history_storage.items():
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            for rnd, val in hist.metrics_distributed.get(metric, []):
                rows.append(
                    {"strategy": name, "round": rnd, "metric": metric, "value": val}
                )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison_metrics.csv"), index=False)

    # -------------------------------------------------------------------
    # Convergence speed: rounds to reach 95% of final F1
    # -------------------------------------------------------------------
    speed_rows = []
    for name in SCENARIOS:
        f1 = df[(df["strategy"] == name) & (df["metric"] == "f1_score")]
        f1 = f1.sort_values("round")
        if len(f1) == 0:
            continue
        final = f1["value"].iloc[-10:].mean()  # mean of last 10 rounds
        target = 0.95 * final
        reached = f1[f1["value"] >= target]
        speed_rows.append(
            {
                "strategy": name,
                "final_f1_mean_last10": round(float(final), 4),
                "rounds_to_95pct": int(reached["round"].iloc[0]) if len(reached) else None,
            }
        )
    pd.DataFrame(speed_rows).to_csv(
        os.path.join(out_dir, "convergence_speed.csv"), index=False
    )

    # -------------------------------------------------------------------
    # 2x2 plot
    # -------------------------------------------------------------------
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics_to_plot):
        for name, cfg in SCENARIOS.items():
            data = history_storage[name].metrics_distributed.get(metric, [])
            if data:
                rounds, values = zip(*data)
                ax.plot(
                    rounds, values, label=name,
                    linestyle=cfg["linestyle"], color=cfg["color"],
                    marker=".", markersize=3, alpha=0.9,
                )
        ax.set_title(metric.replace("_", " ").capitalize(), fontsize=13)
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
    for ax in axes[2:]:
        ax.set_xlabel("Federated Round", fontsize=11)
    fig.suptitle(
        "FedCVR vs. FedAvg – Convergence Comparison (100 rounds, 5 clients)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "comparison_plot.png"), dpi=150,
                bbox_inches="tight")
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedCVR – convergence comparison")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.data_dir, args.rounds, args.out_dir, args.seed)
