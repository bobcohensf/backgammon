"""
Backgammon board representation and state management.

Board layout (from Player 1's perspective):
13 14 15 16 17 18    19 20 21 22 23 24
                BAR
12 11 10  9  8  7     6  5  4  3  2  1

Points 1-6: Player 1's home board
Points 7-12: Player 1's outer board
Points 13-18: Player 2's outer board
Points 19-24: Player 2's home board

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

    def copy(self):
        """Create a deep copy of the board."""
        new_board = Board(self.points.copy())
        new_board.bar = self.bar.copy()
        new_board.off = self.off.copy()
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
            # Home board is points 1-6
            for point in range(7, 25):
                if self.points[point] > 0:
                    return False
        else:
            # Home board is points 19-24
            for point in range(1, 19):
                if self.points[point] < 0:
                    return False

        return True

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

        # For each point, encode: how many of our pieces, how many opponent pieces
        # We'll use a more sophisticated encoding with multiple features per point
        for point in range(1, 25):
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
