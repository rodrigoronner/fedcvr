"""
model.py – Deep Neural Network for cardiovascular risk binary classification.

Architecture (as described in Section 3.2 of the paper)
--------------------------------------------------------
Input(13) → Linear(13, 64) → ReLU → Dropout(0.3)
          → Linear(64, 32) → ReLU → Dropout(0.3)
          → Linear(32, 1)  → Sigmoid

The network outputs a probability in [0, 1] and is trained with
``torch.nn.BCELoss`` (binary cross-entropy), matching the manuscript.
Linear layers are initialised with Xavier uniform initialisation, as
stated in the Implementation Details section of the paper.
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """Fully connected network returning sigmoid probabilities.

    Parameters
    ----------
    input_features : int
        Number of input features. The harmonised cardiovascular feature
        set used in this project contains 13 features.
    dropout : float
        Dropout probability applied after each hidden layer (default 0.3).
    """

    def __init__(self, input_features: int = 13, dropout: float = 0.3) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.drop1 = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(p=dropout)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # Xavier initialisation (Implementation Details, Section 3.5)
        for layer in (self.layer1, self.layer2, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = self.drop1(x)
        x = torch.relu(self.layer2(x))
        x = self.drop2(x)
        return self.sigmoid(self.output(x))  # probability in [0, 1]
