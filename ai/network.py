"""
Neural network for backgammon position evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MatchAwareBackgammonNet(nn.Module):
    """Neural network for evaluating backgammon positions with match context.

    Takes board state encoding + match context as input and outputs:
    - Match equity (probability of winning the match from this position)
    - Cube decision values (double/take/drop probabilities)
    """

    def __init__(self, board_input_size=196, hidden_sizes=[256, 128, 64]):
        """Initialize the match-aware network.

        Args:
            board_input_size: Size of board feature vector (default 196)
            hidden_sizes: List of hidden layer sizes
        """
        super(MatchAwareBackgammonNet, self).__init__()

        # Match context: 5 additional features
        # [my_score / match_length, opp_score / match_length, cube_value / 64,
        #  is_crawford, cube_owner_indicator]
        self.match_context_size = 5
        input_size = board_input_size + self.match_context_size

        # Shared layers for feature extraction
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        self.shared_network = nn.Sequential(*layers)

        # Output heads
        # 1. Match equity head (probability of winning match)
        self.match_equity_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 2. Cube decision head (3 outputs: double value, take/drop threshold)
        self.cube_decision_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [should_double_score, should_accept_score]
            nn.Sigmoid()
        )

    def forward(self, board_features, match_context):
        """Forward pass.

        Args:
            board_features: Tensor of board features [batch, 196]
            match_context: Tensor of match context [batch, 5]

        Returns:
            Dictionary with 'match_equity' and 'cube_decisions'
        """
        # Concatenate board and match features
        x = torch.cat([board_features, match_context], dim=1)

        # Shared feature extraction
        shared = self.shared_network(x)

        # Compute outputs
        match_equity = self.match_equity_head(shared)
        cube_decisions = self.cube_decision_head(shared)

        return {
            'match_equity': match_equity,
            'cube_decisions': cube_decisions
        }

    def evaluate_match_equity(self, board, player, my_score, opp_score,
                             match_length, is_crawford=False):
        """Evaluate match equity for a position.

        Args:
            board: Board object
            player: Player perspective (1 or -1)
            my_score: Current player's match score
            opp_score: Opponent's match score
            match_length: Match length (points to win)
            is_crawford: Whether this is Crawford game

        Returns:
            Match equity (probability of winning match from this position)
        """
        board_features = board.encode_for_nn(player)
        match_context = self._encode_match_context(
            board, player, my_score, opp_score, match_length, is_crawford
        )

        with torch.no_grad():
            board_tensor = torch.from_numpy(board_features).float().unsqueeze(0)
            context_tensor = torch.from_numpy(match_context).float().unsqueeze(0)
            output = self.forward(board_tensor, context_tensor)
            return output['match_equity'].item()

    def evaluate_cube_decision(self, board, player, my_score, opp_score,
                               match_length, is_crawford=False):
        """Evaluate cube decisions.

        Args:
            board: Board object
            player: Player perspective
            my_score, opp_score: Match scores
            match_length: Match length
            is_crawford: Crawford game indicator

        Returns:
            Dictionary with 'should_double' and 'should_accept' probabilities
        """
        board_features = board.encode_for_nn(player)
        match_context = self._encode_match_context(
            board, player, my_score, opp_score, match_length, is_crawford
        )

        with torch.no_grad():
            board_tensor = torch.from_numpy(board_features).float().unsqueeze(0)
            context_tensor = torch.from_numpy(match_context).float().unsqueeze(0)
            output = self.forward(board_tensor, context_tensor)
            cube_decisions = output['cube_decisions'][0]

            return {
                'should_double': cube_decisions[0].item(),
                'should_accept': cube_decisions[1].item()
            }

    def _encode_match_context(self, board, player, my_score, opp_score,
                              match_length, is_crawford):
        """Encode match context as features.

        Returns:
            Numpy array of match context features [5]
        """
        # Normalize scores
        my_score_norm = my_score / match_length
        opp_score_norm = opp_score / match_length

        # Normalize cube value (max 64)
        cube_value_norm = board.cube_value / 64.0

        # Crawford indicator (0 or 1)
        crawford_indicator = 1.0 if is_crawford else 0.0

        # Cube owner indicator (-1, 0, 1 -> normalized)
        if board.cube_owner is None:
            cube_owner_indicator = 0.0
        elif board.cube_owner == player:
            cube_owner_indicator = 1.0
        else:
            cube_owner_indicator = -1.0

        return np.array([
            my_score_norm,
            opp_score_norm,
            cube_value_norm,
            crawford_indicator,
            cube_owner_indicator
        ], dtype=np.float32)

    def save(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model weights."""
        self.load_state_dict(torch.load(path))
        self.eval()


class BackgammonNet(nn.Module):
    """Neural network for evaluating backgammon positions.

    Takes board state encoding as input and outputs win probability
    for the player from whose perspective the board is encoded.
    """

    def __init__(self, input_size=196, hidden_sizes=[256, 128, 64]):
        """Initialize the network.

        Args:
            input_size: Size of input feature vector (default 196 from board encoding)
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
