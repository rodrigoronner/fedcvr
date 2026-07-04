"""
fedcvr/model.py
===============
Deep Neural Network for binary cardiovascular risk classification.

Architecture (Section 3.2 of the paper)
-----------------------------------------
Input(13) -> Linear(13, 64) -> ReLU -> Dropout(0.3)
          -> Linear(64, 32) -> ReLU -> Dropout(0.3)
          -> Linear(32, 1)  -> Sigmoid

The network outputs a probability in [0, 1] and is trained with
torch.nn.BCELoss (binary cross-entropy on sigmoid probabilities),
as described in Section 3.5 of the paper.
All linear layers are initialized with Xavier uniform initialization.

Authors: Rodrigo Tertulino, Ricardo Almeida, Laercio Alencar
IFRN - Federal Institute of Education, Science and Technology of
Rio Grande do Norte, Mossoró, RN, Brazil.
Repository: https://github.com/rodrigoronner/fedcvr
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """Fully-connected network returning sigmoid probabilities.

    Parameters
    ----------
    input_features : int
        Number of input features.  The harmonized 13-attribute UCI
        Heart Disease schema is used throughout the paper.
    dropout : float
        Dropout probability applied after each hidden layer (default 0.3).
    """

    def __init__(self, input_features: int = 13, dropout: float = 0.3) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.drop1  = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(64, 32)
        self.drop2  = nn.Dropout(p=dropout)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        for layer in (self.layer1, self.layer2, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = self.drop1(x)
        x = torch.relu(self.layer2(x))
        x = self.drop2(x)
        return self.sigmoid(self.output(x))
