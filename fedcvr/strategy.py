"""
fedcvr/strategy.py
==================
FedCVR server-side adaptive aggregation strategy.

Implements the bias-corrected Adam-style server optimizer described in
Section 3.1 of the paper (Equations 3 to 6):

    Delta_t = FedAvg(client_parameters) - w_t          (Eq. 3, pseudo-gradient)
    m_t     = beta_1 * m_{t-1} + (1 - beta_1) * Delta_t  (Eq. 4, 1st moment)
    v_t     = beta_2 * v_{t-1} + (1 - beta_2) * Delta_t^2 (Eq. 4, 2nd moment)
    m_hat_t = m_t / (1 - beta_1^t)                     (Eq. 5, bias correction)
    v_hat_t = v_t / (1 - beta_2^t)                     (Eq. 5, bias correction)
    w_{t+1} = w_t + eta * m_hat_t / (sqrt(v_hat_t) + eps_opt)  (Eq. 6, update)

Note the + sign in Eq. 6: Delta_t is the improvement direction
(aggregated params minus current params), so the server adds the
Adam-smoothed pseudo-gradient to the current model.

Default hyperparameters (Section 3.5 of the paper):
    eta = 1.0, beta_1 = 0.9, beta_2 = 0.999, eps_opt = 1e-8

Setting eta = 0.0 disables the server optimizer entirely, reducing the
strategy to plain FedAvg (used as the baseline in the experiments).

Authors: Rodrigo Tertulino, Ricardo Almeida, Laercio Alencar
IFRN - Federal Institute of Education, Science and Technology of
Rio Grande do Norte, Mossoró, RN, Brazil.
Repository: https://github.com/rodrigoronner/fedcvr
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedCVRStrategy(FedAvg):
    """FedAvg + Adam-style server optimizer.

    Parameters
    ----------
    eta     : Server learning rate eta (paper Section 3.5: 1.0).
    beta_1  : Exponential decay for the 1st moment (beta_1 = 0.9).
    beta_2  : Exponential decay for the 2nd moment (beta_2 = 0.999).
    eps_opt : Numerical stability constant epsilon_opt (1e-8).
    **kwargs: Forwarded verbatim to FedAvg.__init__.
    """

    def __init__(
        self,
        *,
        eta: float = 1.0,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps_opt: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.eta     = eta
        self.beta_1  = beta_1
        self.beta_2  = beta_2
        self.eps_opt = eps_opt

        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None
        self._current_weights: Optional[List[np.ndarray]] = None
        self._t: int = 0

        self.client_metrics_history: Dict[int, Dict] = {}
        self.final_weights: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is None:
            return aggregated_parameters, aggregated_metrics

        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        # eta = 0.0 -> plain FedAvg baseline
        if self.eta == 0.0:
            self.final_weights = aggregated_parameters
            return aggregated_parameters, aggregated_metrics

        # Bootstrap: first round initializes server state
        if self._current_weights is None:
            self._current_weights = [np.copy(p) for p in aggregated_ndarrays]
            self._m   = [np.zeros_like(p) for p in aggregated_ndarrays]
            self._v   = [np.zeros_like(p) for p in aggregated_ndarrays]
            self.final_weights = ndarrays_to_parameters(self._current_weights)
            return self.final_weights, aggregated_metrics

        # Eq. 3: pseudo-gradient = aggregated params - current params
        delta = [
            agg - cur
            for agg, cur in zip(aggregated_ndarrays, self._current_weights)
        ]

        # Eq. 4: moment updates
        self._t += 1
        t = self._t
        self._m = [
            self.beta_1 * m_prev + (1.0 - self.beta_1) * d
            for m_prev, d in zip(self._m, delta)
        ]
        self._v = [
            self.beta_2 * v_prev + (1.0 - self.beta_2) * (d ** 2)
            for v_prev, d in zip(self._v, delta)
        ]

        # Eq. 5: bias correction
        m_hat = [m / (1.0 - self.beta_1 ** t) for m in self._m]
        v_hat = [v / (1.0 - self.beta_2 ** t) for v in self._v]

        # Eq. 6: server parameter update (+ sign: pseudo-gradient is improvement direction)
        new_weights = [
            w + self.eta * mh / (np.sqrt(vh) + self.eps_opt)
            for w, mh, vh in zip(self._current_weights, m_hat, v_hat)
        ]

        self._current_weights = new_weights
        updated_parameters    = ndarrays_to_parameters(new_weights)
        self.final_weights    = updated_parameters
        return updated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        if results:
            self.client_metrics_history[server_round] = {
                proxy.cid: {"loss": res.loss, **res.metrics}
                for proxy, res in results
            }
        return aggregated_loss, aggregated_metrics
