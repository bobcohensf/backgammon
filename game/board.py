"""
Backgammon board representation and state management.

Board layout:
13 14 15 16 17 18    19 20 21 22 23 24
                BAR
12 11 10  9  8  7     6  5  4  3  2  1

Points 1-6: Player 2's home board (Player 2 bears off here)
Points 7-12: Player 2's outer board
Points 13-18: Player 1's outer board
Points 19-24: Player 1's home board (Player 1 bears off here)

Player 1 moves UP: 1→2→...→24→OFF
Player 2 moves DOWN: 24→23→...→1→OFF

Positive numbers = Player 1's pieces
Negative numbers = Player 2's pieces
"""

import numpy as np
from typing import List, Tuple, Optional
import copy


class Board:
    """Represents a backgammon board state."""

    # Standard starting position
    INITIAL_POSITION = [
        0,   # Point 0 (unused, for 1-indexed points)
        2,   # Point 1
        0, 0, 0, 0, -5,  # Points 2-6
        0, -3,  # Points 7-8
        0, 0, 0, 5,  # Points 9-12
        -5,  # Point 13
        0, 0, 0, 3,  # Points 14-17
        0, 5,  # Points 18-19
        0, 0, 0, 0, -2  # Points 20-24
    ]

    def __init__(self, position=None):
        """Initialize a board.

        Args:
            position: Optional custom position. If None, uses standard starting position.
        """
        if position is None:
            self.points = self.INITIAL_POSITION.copy()
        else:
            self.points = position.copy()

        # Bar: pieces that have been hit
        self.bar = {1: 0, -1: 0}  # Player 1 and Player 2

        # Off: pieces that have been borne off
        self.off = {1: 0, -1: 0}

        # Doubling cube state
        self.cube_value = 1  # Current stake (1, 2, 4, 8, 16, 32, 64)
        self.cube_owner = None  # None (center), 1, or -1 (who can double next)
        self.double_offered_by = None  # Track if a double is currently offered

    def copy(self):
        """Create a deep copy of the board."""
        new_board = Board(self.points.copy())
        new_board.bar = self.bar.copy()
        new_board.off = self.off.copy()
        new_board.cube_value = self.cube_value
        new_board.cube_owner = self.cube_owner
        new_board.double_offered_by = self.double_offered_by
        return new_board

    def get_piece_count(self, point: int, player: int) -> int:
        """Get number of pieces for a player at a point.

        Args:
            point: Point number (1-24)
            player: 1 or -1

        Returns:
            Number of pieces (0 or positive)
        """
        if point < 1 or point > 24:
            return 0
        value = self.points[point]
        if (player == 1 and value > 0) or (player == -1 and value < 0):
            return abs(value)
        return 0

    def can_enter_from_bar(self, player: int, point: int) -> bool:
        """Check if a player can enter from the bar to a point.

        Args:
            player: 1 or -1
            point: Point to enter to

        Returns:
            True if the move is legal
        """
        if point < 1 or point > 24:
            return False

        opponent = -player
        # Can enter if point is empty, owned by player, or has only 1 opponent piece
        value = self.points[point]

        if value == 0:
            return True
        if (player == 1 and value > 0) or (player == -1 and value < 0):
            return True
        # Can hit a blot (single opponent piece)
        if abs(value) == 1:
            return True
        return False

    def can_move(self, player: int, from_point: int, to_point: int) -> bool:
        """Check if a move is legal.

        Args:
            player: 1 or -1
            from_point: Starting point (1-24, or 0 for bar, or 25/0 for bearing off)
            to_point: Ending point

        Returns:
            True if the move is legal
        """
        # Check if from_point has player's pieces
        if from_point == 0 or from_point == 25:
            # Bar or bearing off
            if from_point == 0 or from_point == 25:
                return False  # Handled separately
        elif from_point < 1 or from_point > 24:
            return False
        else:
            value = self.points[from_point]
            if (player == 1 and value <= 0) or (player == -1 and value >= 0):
                return False

        # Check if to_point is valid for landing
        if to_point < 0 or to_point > 25:
            return False

        if to_point == 0 or to_point == 25:
            # Bearing off - check later
            return True

        opponent_value = self.points[to_point]
        opponent = -player

        # Can't land on opponent's point with 2+ pieces
        if (opponent == 1 and opponent_value >= 2) or (opponent == -1 and opponent_value <= -2):
            return False

        return True

    def apply_move(self, player: int, from_point: int, to_point: int):
        """Apply a move to the board (modifies in place).

        Args:
            player: 1 or -1
            from_point: Starting point (0 for bar, 1-24 for points)
            to_point: Ending point (1-24 for points, 25/0 for bearing off)
        """
        # Remove piece from origin
        if from_point == 0:
            # Moving from bar
            self.bar[player] -= 1
        elif 1 <= from_point <= 24:
            if player == 1:
                self.points[from_point] -= 1
            else:
                self.points[from_point] += 1

        # Place piece at destination (or bear off)
        if to_point == 0 or to_point == 25:
            # Bearing off
            self.off[player] += 1
        elif 1 <= to_point <= 24:
            opponent = -player
            # Check if we're hitting an opponent's blot
            value = self.points[to_point]
            if (opponent == 1 and value == 1) or (opponent == -1 and value == -1):
                # Hit the blot
                self.points[to_point] = 0
                self.bar[opponent] += 1

            # Place our piece
            if player == 1:
                self.points[to_point] += 1
            else:
                self.points[to_point] -= 1

    def is_game_over(self) -> Optional[int]:
        """Check if the game is over.

        Returns:
            Winner (1 or -1) if game is over, None otherwise
        """
        if self.off[1] == 15:
            return 1
        if self.off[-1] == 15:
            return -1
        return None

    def can_bear_off(self, player: int) -> bool:
        """Check if a player can bear off (all pieces in home board).

        Args:
            player: 1 or -1

        Returns:
            True if player can bear off
        """
        # Must have no pieces on bar
        if self.bar[player] > 0:
            return False

        # All pieces must be in home board
        if player == 1:
            # Home board is points 19-24
            for point in range(1, 19):
                if self.points[point] > 0:
                    return False
        else:
            # Home board is points 1-6
            for point in range(7, 25):
                if self.points[point] < 0:
                    return False

        return True

    def can_offer_double(self, player: int) -> bool:
        """Check if a player can offer a double.

        Args:
            player: 1 or -1

        Returns:
            True if player can double
        """
        # Can't double if cube is at maximum (64)
        if self.cube_value >= 64:
            return False

        # Can double if cube is in center or if player owns it
        return self.cube_owner is None or self.cube_owner == player

    def offer_double(self, player: int):
        """Offer a double to the opponent.

        Args:
            player: Player offering the double (1 or -1)
        """
        if not self.can_offer_double(player):
            raise ValueError(f"Player {player} cannot offer a double")

        self.double_offered_by = player

    def accept_double(self, player: int):
        """Accept a double offer.

        Args:
            player: Player accepting the double (1 or -1)
        """
        if self.double_offered_by is None:
            raise ValueError("No double has been offered")

        opponent = -player
        if self.double_offered_by != opponent:
            raise ValueError("Only the opponent's double can be accepted")

        # Double the cube value and give ownership to accepting player
        self.cube_value *= 2
        self.cube_owner = player
        self.double_offered_by = None

    def reject_double(self):
        """Reject a double offer (forfeit the game)."""
        if self.double_offered_by is None:
            raise ValueError("No double has been offered")

        self.double_offered_by = None

    def is_gammon(self, winner: int) -> bool:
        """Check if the win is a gammon (opponent has borne off no pieces).

        Args:
            winner: The winning player (1 or -1)

        Returns:
            True if this is a gammon
        """
        loser = -winner
        return self.off[loser] == 0

    def is_backgammon(self, winner: int) -> bool:
        """Check if the win is a backgammon (opponent has pieces in winner's home or on bar).

        Args:
            winner: The winning player (1 or -1)

        Returns:
            True if this is a backgammon
        """
        if not self.is_gammon(winner):
            return False

        loser = -winner

        # Check if loser has pieces on bar
        if self.bar[loser] > 0:
            return True

        # Check if loser has pieces in winner's home board
        if winner == 1:
            # Winner is player 1, home is 19-24
            # Check if loser (player -1) has pieces in points 19-24
            for point in range(19, 25):
                if self.points[point] < 0:
                    return True
        else:
            # Winner is player -1, home is 1-6
            # Check if loser (player 1) has pieces in points 1-6
            for point in range(1, 7):
                if self.points[point] > 0:
                    return True

        return False

    def get_points_for_win(self, winner: int) -> int:
        """Calculate points scored for a win based on game type and cube value.

        Args:
            winner: The winning player (1 or -1)

        Returns:
            Points scored (cube_value × multiplier)
        """
        if self.is_backgammon(winner):
            multiplier = 3
        elif self.is_gammon(winner):
            multiplier = 2
        else:
            multiplier = 1

        return self.cube_value * multiplier

    def encode_for_nn(self, player: int) -> np.ndarray:
        """Encode board state for neural network input.

        Creates a feature vector representing the board from the perspective
        of the given player.

        Args:
            player: 1 or -1 (perspective)

        Returns:
            Numpy array of features
        """
        features = []

        # Encode board from current player's perspective
        # Always encode from opponent's home toward their own home
        # This makes the spatial relationships consistent for both players

        if player == 1:
            # Player 1: home is 19-24, moves from 1→24
            # Encode in order that matches their movement: 1→2→...→23→24
            point_order = range(1, 25)
        else:
            # Player 2: home is 1-6, moves from 24→1
            # Encode in order that matches their movement: 24→23→...→2→1
            point_order = range(24, 0, -1)

        # For each point, encode: how many of our pieces, how many opponent pieces
        # We'll use a more sophisticated encoding with multiple features per point
        for point in point_order:
            value = self.points[point]

            if player == 1:
                our_pieces = max(0, value)
                opp_pieces = max(0, -value)
            else:
                our_pieces = max(0, -value)
                opp_pieces = max(0, value)

            # Encode piece counts (capped at 3+ for compactness)
            features.append(1.0 if our_pieces >= 1 else 0.0)
            features.append(1.0 if our_pieces >= 2 else 0.0)
            features.append(1.0 if our_pieces >= 3 else 0.0)
            features.append(float(max(0, our_pieces - 3)) / 2.0)  # Normalized excess

            features.append(1.0 if opp_pieces >= 1 else 0.0)
            features.append(1.0 if opp_pieces >= 2 else 0.0)
            features.append(1.0 if opp_pieces >= 3 else 0.0)
            features.append(float(max(0, opp_pieces - 3)) / 2.0)

        # Bar pieces
        features.append(float(self.bar[player]) / 2.0)
        features.append(float(self.bar[-player]) / 2.0)

        # Off pieces
        features.append(float(self.off[player]) / 15.0)
        features.append(float(self.off[-player]) / 15.0)

        return np.array(features, dtype=np.float32)

    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        lines.append("=" * 50)

        # Top half (points 13-24)
        top_points = []
        for i in range(13, 25):
            val = self.points[i]
            if val == 0:
                top_points.append("  .  ")
            else:
                player = "X" if val > 0 else "O"
                count = abs(val)
                top_points.append(f" {player}{count:2d} ")

        # Split into two halves with BAR in middle
        top_left = " ".join(top_points[0:6])
        top_right = " ".join(top_points[6:12])
        lines.append(f"{top_left} | {top_right}")

        # Bar
        bar_display = f"BAR: X:{self.bar[1]} O:{self.bar[-1]}"
        lines.append(bar_display.center(50))

        # Bottom half (points 12-1)
        bottom_points = []
        for i in range(12, 0, -1):
            val = self.points[i]
            if val == 0:
                bottom_points.append("  .  ")
            else:
                player = "X" if val > 0 else "O"
                count = abs(val)
                bottom_points.append(f" {player}{count:2d} ")

        bottom_left = " ".join(bottom_points[0:6])
        bottom_right = " ".join(bottom_points[6:12])
        lines.append(f"{bottom_left} | {bottom_right}")

        # Off
        lines.append(f"OFF: X:{self.off[1]} O:{self.off[-1]}")
        lines.append("=" * 50)

        return "\n".join(lines)
