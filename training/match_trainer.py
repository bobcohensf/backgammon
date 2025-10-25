"""
Match-play training with TD(λ) learning.

This trainer plays full matches and optimizes for match equity
rather than single-game wins. It also trains cube decisions.
"""

import torch
import numpy as np
from tqdm import tqdm
import os
from typing import List, Tuple, Optional
from game import BackgammonGame, Board
from ai.agent import MatchAwareTDAgent


class MatchPlayTrainer:
    """Trainer for match-play with cube decisions."""

    def __init__(self, agent: MatchAwareTDAgent, match_length=7, save_dir='models'):
        """Initialize match play trainer.

        Args:
            agent: MatchAwareTDAgent to train
            match_length: Points to win match (default 7)
            save_dir: Directory to save model checkpoints
        """
        self.agent = agent
        self.match_length = match_length
        self.agent.match_length = match_length
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Training statistics
        self.matches_played = 0
        self.match_wins = {1: 0, -1: 0}
        self.games_played = 0
        self.games_won = {1: 0, -1: 0}
        self.doubles_offered = 0
        self.doubles_accepted = 0
        self.doubles_rejected = 0

    def play_match(self, training=True) -> int:
        """Play one complete match.

        Args:
            training: If True, perform TD learning updates

        Returns:
            Match winner (1 or -1)
        """
        match_score = {1: 0, -1: 0}
        crawford_game = False
        post_crawford = False

        # Track match-level states for training
        match_states = []  # List of (board, player, my_score, opp_score, crawford) tuples
        match_equities = []  # List of match equity tensors

        while match_score[1] < self.match_length and match_score[-1] < self.match_length:
            # Check Crawford rule
            if not crawford_game and not post_crawford:
                if match_score[1] == self.match_length - 1 or match_score[-1] == self.match_length - 1:
                    crawford_game = True

            # Play one game
            game_result = self.play_game(
                match_score, crawford_game, training, match_states, match_equities
            )

            winner = game_result['winner']
            points = game_result['points']

            # Update match score
            match_score[winner] += points
            self.games_won[winner] += 1
            self.games_played += 1

            # Update Crawford status
            if crawford_game:
                crawford_game = False
                post_crawford = True

        # Match complete - determine winner
        match_winner = 1 if match_score[1] >= self.match_length else -1
        self.match_wins[match_winner] += 1
        self.matches_played += 1

        # Perform match-level TD updates
        if training and len(match_states) > 0:
            # Terminal match equity is 1.0 for winner, 0.0 for loser
            self._match_td_update(match_states, match_equities, match_winner)

        return match_winner

    def play_game(self, match_score: dict, crawford_game: bool, training: bool,
                  match_states: List, match_equities: List) -> dict:
        """Play one game within a match.

        Args:
            match_score: Current match score {1: score, -1: score}
            crawford_game: Whether this is Crawford game
            training: Whether to collect training data
            match_states: List to accumulate match states
            match_equities: List to accumulate equity evaluations

        Returns:
            Dictionary with game result info
        """
        game = BackgammonGame()
        move_count = 0
        max_moves = 2000

        while move_count < max_moves:
            player = game.current_player
            my_score = match_score[player]
            opp_score = match_score[-player]

            # Check if should offer double (before rolling dice)
            if not crawford_game and move_count > 0:  # Don't double on first move
                should_double = self.agent.should_offer_double(
                    game.board, player, my_score, opp_score, crawford_game
                )

                if should_double:
                    self.doubles_offered += 1

                    # Opponent decides whether to accept
                    opponent = -player
                    opp_my_score = match_score[opponent]
                    opp_opp_score = match_score[player]

                    should_accept = self.agent.should_accept_double(
                        game.board, opponent, opp_my_score, opp_opp_score, crawford_game
                    )

                    if should_accept:
                        # Accept double
                        self.doubles_accepted += 1
                        game.board.cube_value *= 2
                        game.board.cube_owner = opponent
                    else:
                        # Reject double - player wins current cube value
                        self.doubles_rejected += 1
                        points = game.board.cube_value
                        return {'winner': player, 'points': points, 'double_rejected': True}

            # Record state for training (before move)
            if training:
                board_copy = game.board.copy()
                match_states.append((
                    board_copy, player, my_score, opp_score, crawford_game
                ))
                equity = self._get_match_equity_tensor(
                    board_copy, player, my_score, opp_score, crawford_game
                )
                match_equities.append(equity)

            # Roll dice
            game.roll_dice()

            # Select and apply move
            move_seq = self.agent.select_move(
                game, player, game.dice, my_score, opp_score, crawford_game, greedy=False
            )
            game.apply_move_sequence(player, move_seq)

            # Check for game over
            winner = game.is_game_over()
            if winner is not None:
                # Calculate points based on game type and cube
                points = game.board.get_points_for_win(winner)

                if training:
                    # Add terminal state
                    match_states.append((
                        game.board.copy(), winner, match_score[winner],
                        match_score[-winner], crawford_game
                    ))
                    # Terminal equity: 1.0 for winner, 0.0 for loser
                    terminal_equity = torch.tensor([[1.0 if winner == player else 0.0]],
                                                  dtype=torch.float32)
                    match_equities.append(terminal_equity)

                return {'winner': winner, 'points': points, 'double_rejected': False}

            game.switch_player()
            move_count += 1

        # Game didn't finish (rare but possible during early training)
        # Assign winner based on progress
        p1_off = game.board.off[1]
        p2_off = game.board.off[-1]
        winner = 1 if p1_off > p2_off else -1
        points = 1  # Assign 1 point for incomplete game

        return {'winner': winner, 'points': points, 'double_rejected': False}

    def _get_match_equity_tensor(self, board: Board, player: int, my_score: int,
                                 opp_score: int, crawford: bool) -> torch.Tensor:
        """Get match equity as a tensor for training.

        Args:
            board: Board state
            player: Player perspective
            my_score, opp_score: Match scores
            crawford: Crawford indicator

        Returns:
            Match equity tensor
        """
        board_features = board.encode_for_nn(player)
        match_context = self.agent.network._encode_match_context(
            board, player, my_score, opp_score, self.match_length, crawford
        )

        board_tensor = torch.from_numpy(board_features).float().unsqueeze(0)
        context_tensor = torch.from_numpy(match_context).float().unsqueeze(0)

        output = self.agent.network.forward(board_tensor, context_tensor)
        return output['match_equity']

    def _match_td_update(self, states: List[Tuple], equities: List[torch.Tensor],
                        match_winner: int):
        """Perform TD updates for a match trajectory.

        Args:
            states: List of (board, player, my_score, opp_score, crawford) tuples
            equities: List of match equity predictions
            match_winner: Winner of the match
        """
        if len(states) == 0:
            return

        # Compute target match equity for each state
        # The target is the match equity from the next state
        total_loss = torch.tensor(0.0, dtype=torch.float32)

        for t in range(len(equities) - 1):
            curr_equity = equities[t]
            next_equity = equities[t + 1].detach()

            # TD error
            td_error = next_equity - curr_equity

            # Accumulate squared TD error
            total_loss = total_loss + (td_error ** 2)

        # Add final state target (actual match outcome)
        if len(equities) > 0:
            # Last state's player
            last_player = states[-1][1]
            final_equity = equities[-1]
            target_equity = torch.tensor([[1.0 if match_winner == last_player else 0.0]],
                                        dtype=torch.float32)
            final_error = target_equity - final_equity
            total_loss = total_loss + (final_error ** 2)

        # Backpropagate
        if total_loss.requires_grad:
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), max_norm=1.0)
            self.agent.optimizer.step()

    def train(self, num_matches=1000, save_every=100, verbose=True):
        """Train the agent through match play.

        Args:
            num_matches: Number of matches to play
            save_every: Save checkpoint every N matches
            verbose: Print progress
        """
        if verbose:
            print(f"Starting match play training for {num_matches} matches...")
            print(f"Match length: {self.match_length} points")
            print(f"Learning rate: {self.agent.optimizer.param_groups[0]['lr']}")
            print(f"Epsilon: {self.agent.epsilon}")

        iterator = tqdm(range(num_matches)) if verbose else range(num_matches)

        for i in iterator:
            self.play_match(training=True)

            # Periodic saving and reporting
            if (i + 1) % save_every == 0:
                self.save_checkpoint(f"match_model_{self.matches_played}.pth")

                if verbose:
                    p1_match_win_rate = self.match_wins[1] / self.matches_played
                    doubles_rate = self.doubles_offered / max(1, self.games_played)
                    accept_rate = self.doubles_accepted / max(1, self.doubles_offered)

                    tqdm.write(
                        f"\nMatches: {self.matches_played}, "
                        f"P1 match win rate: {p1_match_win_rate:.3f}\n"
                        f"  Games: {self.games_played}, "
                        f"Doubles/game: {doubles_rate:.2f}, "
                        f"Accept rate: {accept_rate:.2%}"
                    )

        if verbose:
            print(f"\nTraining complete! Total matches: {self.matches_played}")
            print(f"Match win rates - P1: {self.match_wins[1]/self.matches_played:.3f}, "
                  f"P2: {self.match_wins[-1]/self.matches_played:.3f}")
            print(f"Doubling stats - Offered: {self.doubles_offered}, "
                  f"Accepted: {self.doubles_accepted}, "
                  f"Rejected: {self.doubles_rejected}")

        # Save final model
        self.save_checkpoint("match_model_final.pth")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        if hasattr(tqdm, 'write'):
            tqdm.write(f"Saved checkpoint: {path}")

    def evaluate(self, num_matches=100, opponent: Optional[MatchAwareTDAgent] = None):
        """Evaluate agent performance.

        Args:
            num_matches: Number of evaluation matches
            opponent: Optional opponent agent

        Returns:
            Dictionary with evaluation statistics
        """
        if opponent is None:
            opponent = self.agent

        wins = {1: 0, -1: 0}
        original_epsilon = self.agent.epsilon

        # Set to greedy for evaluation
        self.agent.epsilon = 0.0
        if opponent != self.agent:
            opponent_epsilon = opponent.epsilon
            opponent.epsilon = 0.0

        for _ in tqdm(range(num_matches), desc="Evaluating"):
            match_score = {1: 0, -1: 0}
            crawford_game = False
            post_crawford = False

            while match_score[1] < self.match_length and match_score[-1] < self.match_length:
                # Check Crawford
                if not crawford_game and not post_crawford:
                    if match_score[1] == self.match_length - 1 or match_score[-1] == self.match_length - 1:
                        crawford_game = True

                # Play game
                game_result = self.play_game(
                    match_score, crawford_game, training=False, match_states=[], match_equities=[]
                )

                match_score[game_result['winner']] += game_result['points']

                if crawford_game:
                    crawford_game = False
                    post_crawford = True

            match_winner = 1 if match_score[1] >= self.match_length else -1
            wins[match_winner] += 1

        # Restore epsilon
        self.agent.epsilon = original_epsilon
        if opponent != self.agent:
            opponent.epsilon = opponent_epsilon

        return {
            'p1_match_wins': wins[1],
            'p2_match_wins': wins[-1],
            'p1_match_win_rate': wins[1] / num_matches,
            'p2_match_win_rate': wins[-1] / num_matches,
            'total_matches': num_matches
        }
