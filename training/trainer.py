"""
Self-play training with TD(λ) learning.
"""

import torch
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, List
from game import BackgammonGame, Board
from ai import TDAgent


class SelfPlayTrainer:
    """Trainer for self-play TD learning."""

    def __init__(self, agent: TDAgent, save_dir='models'):
        """Initialize trainer.

        Args:
            agent: TDAgent to train
            save_dir: Directory to save model checkpoints
        """
        self.agent = agent
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Training statistics
        self.games_played = 0
        self.wins = {1: 0, -1: 0}
        self.incomplete_games = 0

    def play_game(self, training=True) -> int:
        """Play one game of self-play.

        Args:
            training: If True, perform TD learning updates

        Returns:
            Winner (1 or -1)
        """
        game = BackgammonGame()

        # Track states for TD updates
        states = []  # List of (board, player) tuples
        values = []  # List of value tensors

        move_count = 0
        max_moves = 3000  # Prevent infinite games (high limit for learning phase)

        while move_count < max_moves:
            player = game.current_player

            # Roll dice
            die1, die2 = game.roll_dice()

            # Get and apply move
            move_seq = self.agent.select_move(game, player, game.dice, greedy=False)

            # Record state before move
            if training:
                board_copy = game.board.copy()
                states.append((board_copy, player))
                value = self.agent.get_state_value(board_copy, player)
                values.append(value)

            # Apply move
            game.apply_move_sequence(player, move_seq)

            # Check for game over
            winner = game.is_game_over()
            if winner is not None:
                # Game over
                if training:
                    # Add terminal state
                    states.append((game.board.copy(), player))
                    terminal_value = torch.tensor([[1.0 if winner == player else 0.0]],
                                                 dtype=torch.float32)
                    values.append(terminal_value)

                    # Perform TD learning updates
                    self._td_update(states, values)

                self.wins[winner] += 1
                self.games_played += 1
                return winner

            game.switch_player()
            move_count += 1

        # Max moves reached - game didn't finish
        # This can happen during early training when network hasn't learned
        self.incomplete_games += 1
        self.games_played += 1

        # Assign winner based on progress (pieces borne off)
        p1_off = game.board.off[1]
        p2_off = game.board.off[-1]

        if p1_off > p2_off:
            winner = 1
        elif p2_off > p1_off:
            winner = -1
        else:
            winner = 1  # Tie goes to player 1

        # IMPORTANT: Perform TD updates even for incomplete games
        # Use a reward based on progress rather than binary win/loss
        if training and len(states) > 0:
            # Calculate reward based on pieces borne off (0.0 to 1.0 scale)
            # The player whose turn it was last gets evaluated
            last_player = states[-1][1]
            player_off = game.board.off[last_player]
            opponent_off = game.board.off[-last_player]

            # Progress-based reward: (our pieces off - opponent pieces off) / 15
            # Normalized to roughly 0-1 range
            progress_reward = (player_off - opponent_off) / 15.0
            # Clamp to [-1, 1] and shift to [0, 1]
            terminal_reward = max(0.0, min(1.0, 0.5 + progress_reward))

            states.append((game.board.copy(), last_player))
            terminal_value = torch.tensor([[terminal_reward]], dtype=torch.float32)
            values.append(terminal_value)

            # Perform TD learning updates
            self._td_update(states, values)

        self.wins[winner] += 1
        return winner

    def _td_update(self, states: List[tuple], values: List[torch.Tensor]):
        """Perform TD(λ) updates for a game trajectory.

        Args:
            states: List of (board, player) tuples
            values: List of value predictions
        """
        # Perform TD updates backwards through the game
        for t in range(len(values) - 1):
            curr_value = values[t]
            next_value = values[t + 1].detach()  # Don't backprop through next value

            # TD error
            td_error = next_value - curr_value

            # Update using gradient descent
            self.agent.optimizer.zero_grad()
            loss = td_error ** 2
            loss.backward()
            self.agent.optimizer.step()

    def train(self, num_games=10000, save_every=1000, verbose=True):
        """Train the agent through self-play.

        Args:
            num_games: Number of games to play
            save_every: Save checkpoint every N games
            verbose: Print progress
        """
        if verbose:
            print(f"Starting training for {num_games} games...")
            print(f"Learning rate: {self.agent.optimizer.param_groups[0]['lr']}")
            print(f"Lambda: {self.agent.lambda_param}")
            print(f"Epsilon: {self.agent.epsilon}")

        iterator = tqdm(range(num_games)) if verbose else range(num_games)

        for i in iterator:
            self.play_game(training=True)

            # Periodic saving and reporting
            if (i + 1) % save_every == 0:
                self.save_checkpoint(f"model_game_{self.games_played}.pth")

                if verbose:
                    win_rate_p1 = self.wins[1] / self.games_played if self.games_played > 0 else 0
                    incomplete_pct = 100 * self.incomplete_games / self.games_played if self.games_played > 0 else 0
                    print(f"\nGames: {self.games_played}, P1 win rate: {win_rate_p1:.3f}, "
                          f"Incomplete: {incomplete_pct:.1f}%")

        if verbose:
            print(f"\nTraining complete! Total games: {self.games_played}")
            print(f"Final win rates - P1: {self.wins[1]/self.games_played:.3f}, "
                  f"P2: {self.wins[-1]/self.games_played:.3f}")
            print(f"Incomplete games: {self.incomplete_games} "
                  f"({100*self.incomplete_games/self.games_played:.1f}%)")

        # Save final model
        self.save_checkpoint("model_final.pth")

    def save_checkpoint(self, filename):
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        if hasattr(tqdm, 'write'):
            tqdm.write(f"Saved checkpoint: {path}")

    def evaluate(self, num_games=100, opponent: Optional[TDAgent] = None):
        """Evaluate agent performance.

        Args:
            num_games: Number of evaluation games
            opponent: Optional opponent agent. If None, plays against self.

        Returns:
            Dictionary with evaluation statistics
        """
        if opponent is None:
            opponent = self.agent

        wins = {1: 0, -1: 0}
        incomplete = 0
        original_epsilon = self.agent.epsilon

        # Set to greedy for evaluation
        self.agent.epsilon = 0.0
        if opponent == self.agent:
            opponent_epsilon = 0.0
        else:
            opponent_epsilon = opponent.epsilon
            opponent.epsilon = 0.0

        for _ in tqdm(range(num_games), desc="Evaluating"):
            game = BackgammonGame()
            move_count = 0
            max_moves = 3000  # Match training limit

            while move_count < max_moves:
                player = game.current_player
                agent = self.agent if player == 1 else opponent

                # Roll and move
                game.roll_dice()
                move_seq = agent.select_move(game, player, game.dice, greedy=True)
                game.apply_move_sequence(player, move_seq)

                winner = game.is_game_over()
                if winner is not None:
                    wins[winner] += 1
                    break

                game.switch_player()
                move_count += 1

            # If game didn't finish, count as incomplete and assign winner based on progress
            if winner is None:
                incomplete += 1
                # Assign win to player with more pieces off
                if game.board.off[1] > game.board.off[-1]:
                    wins[1] += 1
                elif game.board.off[-1] > game.board.off[1]:
                    wins[-1] += 1
                else:
                    wins[1] += 1  # Tie goes to player 1

        # Restore epsilon
        self.agent.epsilon = original_epsilon
        if opponent != self.agent:
            opponent.epsilon = opponent_epsilon

        return {
            'p1_wins': wins[1],
            'p2_wins': wins[-1],
            'p1_win_rate': wins[1] / num_games,
            'p2_win_rate': wins[-1] / num_games,
            'total_games': num_games,
            'incomplete': incomplete,
            'incomplete_rate': incomplete / num_games
        }
