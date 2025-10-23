"""
Neural network for backgammon position evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BackgammonNet(nn.Module):
    """Neural network for evaluating backgammon positions.

    Takes board state encoding as input and outputs win probability
    for the player from whose perspective the board is encoded.
    """

    def __init__(self, input_size=198, hidden_sizes=[256, 128, 64]):
        """Initialize the network.

        Args:
            input_size: Size of input feature vector (default 198 from board encoding)
            hidden_sizes: List of hidden layer sizes
        """
        super(BackgammonNet, self).__init__()

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer: single value for win probability
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of board features

        Returns:
            Win probability (0-1)
        """
        return self.network(x)

    def evaluate(self, board, player):
        """Evaluate a board position for a player.

        Args:
            board: Board object
            player: Player (1 or -1)

        Returns:
            Win probability for the player (0-1)
        """
        features = board.encode_for_nn(player)
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)
            prob = self.forward(x).item()
        return prob

    def save(self, path):
        """Save model weights.

        Args:
            path: File path to save to
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model weights.

        Args:
            path: File path to load from
        """
        self.load_state_dict(torch.load(path))
        self.eval()
