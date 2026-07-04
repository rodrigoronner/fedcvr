"""
fedcvr/client.py
================
FedCVR Flower client implementing local training and client-level
Differential Privacy.

Local training (Section 3.1 of the paper)
------------------------------------------
Each client trains the global model on its private dataset for E=5 local
epochs using the Adam optimizer (lr=0.001) and binary cross-entropy loss
on sigmoid outputs (BCELoss).

Client-level Differential Privacy (Equations 8 and 9)
------------------------------------------------------
After local training, the client computes the update vector:

    Delta_theta = theta_local - theta_global           (Eq. 7)

It then clips the global L2 norm of this vector to C and adds
calibrated Gaussian noise:

    Delta_clip = Delta_theta * min(1, C / ||Delta_theta||_2)  (Eq. 8)
    Delta_noisy = Delta_clip + N(0, sigma^2 * C^2 * I)        (Eq. 9)

The noisy update is sent to the server; the server never receives the
unperturbed update.  This provides client-level DP: an adversary
observing all server-side information cannot reliably determine whether
a given institution participated in a given round.

Authors: Rodrigo Tertulino, Ricardo Almeida, Laercio Alencar
IFRN - Federal Institute of Education, Science and Technology of
Rio Grande do Norte, Mossoró, RN, Brazil.
Repository: https://github.com/rodrigoronner/fedcvr
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import NumPyClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .model import Net


def privatize_update(
    delta: List[np.ndarray],
    max_grad_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Clip and perturb the update vector (Equations 8 and 9).

    Parameters
    ----------
    delta           : Per-layer list of update arrays (theta_local - theta_global).
    max_grad_norm   : Clipping norm C.
    noise_multiplier: sigma; noise standard deviation is sigma * C.
    rng             : Seeded numpy random Generator for reproducibility.
    """
    total_norm = float(np.sqrt(sum(float((d ** 2).sum()) for d in delta)))
    clip_coef  = min(1.0, max_grad_norm / (total_norm + 1e-12))
    clipped    = [d * clip_coef for d in delta]

    noise_std  = noise_multiplier * max_grad_norm
    noisy      = [
        c + rng.normal(loc=0.0, scale=noise_std, size=c.shape).astype(c.dtype)
        for c in clipped
    ]
    return noisy


class FedCVRClient(NumPyClient):
    """Flower NumPyClient implementing FedCVR local training.

    Parameters
    ----------
    model        : Initialized Net instance.
    train_loader : DataLoader for local training data.
    test_loader  : DataLoader for local evaluation data.
    local_epochs : Local SGD epochs per communication round (paper: 5).
    use_dp       : Whether to apply client-level DP to the update.
    dp_config    : Dict with keys noise_multiplier and max_grad_norm.
    seed         : Seed for the DP noise generator.
    """

    def __init__(
        self,
        model: Net,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_epochs: int = 5,
        use_dp: bool = False,
        dp_config: Optional[Dict] = None,
        seed: int = 42,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.local_epochs = local_epochs
        self.use_dp       = use_dp
        self.dp_config    = dp_config or {}
        self._rng         = np.random.default_rng(seed)

        # Section 3.5: Adam at the client, lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Section 3.5: binary cross-entropy on sigmoid outputs
        self.criterion = nn.BCELoss()

        if self.use_dp and not self.dp_config:
            raise ValueError("dp_config must be provided when use_dp=True")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        global_params = [np.copy(p) for p in parameters]

        self.model.train()
        for _ in range(self.local_epochs):
            for features, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        local_params = self.get_parameters({})

        if self.use_dp:
            # Equations 8 and 9: clip update, add Gaussian noise.
            delta      = [lp - gp for lp, gp in zip(local_params, global_params)]
            noisy_delta = privatize_update(
                delta,
                max_grad_norm   = self.dp_config["max_grad_norm"],
                noise_multiplier = self.dp_config["noise_multiplier"],
                rng             = self._rng,
            )
            out_params = [gp + nd for gp, nd in zip(global_params, noisy_delta)]
        else:
            out_params = local_params

        return out_params, len(self.train_loader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        all_labels: List[float] = []
        all_probs:  List[float] = []
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in self.test_loader:
                probs       = self.model(features)
                total_loss += self.criterion(probs, labels).item() * len(labels)
                all_labels.extend(labels.numpy().flatten().tolist())
                all_probs.extend(probs.numpy().flatten().tolist())

        n = len(all_labels)
        if n == 0:
            return 0.0, 0, {}

        y_true   = np.array(all_labels)
        y_prob   = np.array(all_probs)
        y_pred   = (y_prob >= 0.5).astype(int)
        avg_loss = total_loss / n

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")

        return (
            float(avg_loss),
            n,
            {
                "accuracy" : float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score" : float(f1_score(y_true, y_pred, zero_division=0)),
                "auc"      : auc,
            },
        )


def build_client(
    cid: str,
    client_train_data: list,
    client_test_data: list,
    batch_size: int = 32,
    local_epochs: int = 5,
    use_dp: bool = False,
    dp_config: Optional[Dict] = None,
    seed: int = 42,
) -> "FedCVRClient":
    """Construct a FedCVRClient from pre-split numpy arrays."""
    idx      = int(cid)
    X_train, y_train = client_train_data[idx]
    X_test,  y_test  = client_test_data[idx]

    model = Net(input_features=X_train.shape[1])

    def _loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).view(-1, 1),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return FedCVRClient(
        model        = model,
        train_loader = _loader(X_train, y_train, shuffle=True),
        test_loader  = _loader(X_test, y_test),
        local_epochs = local_epochs,
        use_dp       = use_dp,
        dp_config    = dp_config,
        seed         = seed + idx,
    )
