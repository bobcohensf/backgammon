"""
TD-Learning agent for backgammon.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional
from game import BackgammonGame, Board, MoveSequence
from .network import BackgammonNet


class TDAgent:
    """Temporal Difference learning agent for backgammon."""

    def __init__(self, network: Optional[BackgammonNet] = None, learning_rate=0.001,
                 lambda_param=0.7, epsilon=0.1):
        """Initialize the TD agent.

        Args:
            network: Neural network for evaluation. If None, creates new one.
            learning_rate: Learning rate for optimizer
            lambda_param: TD(λ) parameter for eligibility traces
            epsilon: Exploration rate (0 = always greedy, 1 = always random)
        """
        self.network = network if network else BackgammonNet()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.lambda_param = lambda_param
        self.epsilon = epsilon

        # For TD(λ) - eligibility traces
        self.eligibility_traces = None

    def select_move(self, game: BackgammonGame, player: int, dice: List[int],
                    greedy=False) -> MoveSequence:
        """Select a move using epsilon-greedy policy.

        Args:
            game: Current game state
            player: Player to move (1 or -1)
            dice: Dice rolled
            greedy: If True, always select best move (no exploration)

        Returns:
            Selected move sequence
        """
        legal_moves = game.get_legal_moves(player, dice)

        if not legal_moves or (len(legal_moves) == 1 and len(legal_moves[0].moves) == 0):
            # No legal moves or only empty move
            return legal_moves[0] if legal_moves else MoveSequence([])

        # Epsilon-greedy exploration
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal_moves)

        # Greedy: select best move based on evaluation
        best_move = None
        best_value = -float('inf')

        for move_seq in legal_moves:
            # Apply move to temporary board
            temp_board = game.board.copy()
            for move in move_seq.moves:
                temp_board.apply_move(player, move.from_point, move.to_point)

            # Evaluate resulting position
            value = self.evaluate_position(temp_board, player)

            if value > best_value:
                best_value = value
                best_move = move_seq

        return best_move if best_move else legal_moves[0]

    def evaluate_position(self, board: Board, player: int) -> float:
        """Evaluate a board position.

        Args:
            board: Board to evaluate
            player: Player from whose perspective to evaluate

        Returns:
            Evaluation score (0-1, higher is better for player)
        """
        # Check if game is over
        winner = board.is_game_over()
        if winner is not None:
            return 1.0 if winner == player else 0.0

        return self.network.evaluate(board, player)

    def get_state_value(self, board: Board, player: int) -> torch.Tensor:
        """Get state value as a tensor (for training).

        Args:
            board: Board state
            player: Player perspective

        Returns:
            Value tensor
        """
        # Check terminal states
        winner = board.is_game_over()
        if winner is not None:
            value = 1.0 if winner == player else 0.0
            return torch.tensor([[value]], dtype=torch.float32)

        features = board.encode_for_nn(player)
        x = torch.from_numpy(features).float().unsqueeze(0)
        return self.network(x)

    def train_step(self, prev_value: torch.Tensor, curr_value: torch.Tensor):
        """Perform one TD learning update.

        Args:
            prev_value: Value of previous state
            curr_value: Value of current state
        """
        # TD error: difference between successive predictions
        # Target is the current value, prediction is the previous value
        td_error = curr_value - prev_value

        # Backward pass on the previous value
        self.optimizer.zero_grad()
        prev_value.backward()

        # Update weights using TD error
        # Scale gradients by TD error
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad *= td_error.item()

        self.optimizer.step()

    def reset_eligibility_traces(self):
        """Reset eligibility traces for TD(λ)."""
        self.eligibility_traces = [torch.zeros_like(param) for param in self.network.parameters()]

    def update_with_traces(self, td_error: float):
        """Update network using eligibility traces.

        Args:
            td_error: TD error to apply
        """
        with torch.no_grad():
            for param, trace in zip(self.network.parameters(), self.eligibility_traces):
                if param.grad is not None:
                    # Accumulate gradient into eligibility trace
                    trace.mul_(self.lambda_param).add_(param.grad)
                    # Update parameter using trace and TD error
                    param.add_(trace, alpha=td_error * self.optimizer.param_groups[0]['lr'])

    def save(self, path):
        """Save the agent's network.

        Args:
            path: File path to save to
        """
        self.network.save(path)

    def load(self, path):
        """Load the agent's network.

        Args:
            path: File path to load from
        """
        self.network.load(path)
