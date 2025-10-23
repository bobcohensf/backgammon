"""
Backgammon game logic, move generation, and rules.
"""

import random
from typing import List, Tuple, Optional
import copy
from .board import Board


class Move:
    """Represents a single checker movement."""

    def __init__(self, from_point: int, to_point: int, die_value: int):
        """
        Args:
            from_point: Starting point (0 for bar, 1-24 for board points)
            to_point: Ending point (1-24 for board points, 25/0 for bearing off)
            die_value: The die value used for this move
        """
        self.from_point = from_point
        self.to_point = to_point
        self.die_value = die_value

    def __repr__(self):
        return f"Move({self.from_point} -> {self.to_point}, die={self.die_value})"

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_point == other.from_point and
                self.to_point == other.to_point and
                self.die_value == other.die_value)

    def __hash__(self):
        return hash((self.from_point, self.to_point, self.die_value))


class MoveSequence:
    """Represents a complete sequence of moves for a turn."""

    def __init__(self, moves: List[Move]):
        self.moves = moves

    def __repr__(self):
        return f"MoveSequence({self.moves})"

    def __eq__(self, other):
        if not isinstance(other, MoveSequence):
            return False
        return self.moves == other.moves

    def __hash__(self):
        return hash(tuple(self.moves))


class BackgammonGame:
    """Main game logic and move generation."""

    def __init__(self, board: Optional[Board] = None):
        """Initialize a game.

        Args:
            board: Optional starting board. If None, uses standard starting position.
        """
        self.board = board if board else Board()
        self.current_player = 1
        self.dice = []

    def roll_dice(self) -> Tuple[int, int]:
        """Roll two dice.

        Returns:
            Tuple of (die1, die2)
        """
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        self.dice = [die1, die2] if die1 != die2 else [die1] * 4
        return (die1, die2)

    def get_legal_moves(self, player: int, dice: List[int], board: Optional[Board] = None) -> List[MoveSequence]:
        """Get all legal move sequences for given dice.

        Args:
            player: 1 or -1
            dice: List of die values (e.g., [3, 5] or [4, 4, 4, 4] for doubles)
            board: Optional board state. If None, uses current game board.

        Returns:
            List of legal move sequences
        """
        if board is None:
            board = self.board

        # Start recursive generation
        all_sequences = []
        self._generate_move_sequences(player, dice, board, [], all_sequences)

        # If no moves found, return empty sequence
        if not all_sequences:
            return [MoveSequence([])]

        # Filter to keep only sequences that use the maximum number of dice
        max_dice_used = max(len(seq.moves) for seq in all_sequences)
        all_sequences = [seq for seq in all_sequences if len(seq.moves) == max_dice_used]

        # Remove duplicates (same final position)
        unique_sequences = []
        seen_boards = set()

        for seq in all_sequences:
            # Apply moves to get final board state
            test_board = board.copy()
            for move in seq.moves:
                test_board.apply_move(player, move.from_point, move.to_point)

            # Use a simple hash of the board state
            board_hash = self._board_hash(test_board)
            if board_hash not in seen_boards:
                seen_boards.add(board_hash)
                unique_sequences.append(seq)

        return unique_sequences if unique_sequences else [MoveSequence([])]

    def _generate_move_sequences(self, player: int, remaining_dice: List[int],
                                   board: Board, current_sequence: List[Move],
                                   all_sequences: List[MoveSequence]):
        """Recursively generate all possible move sequences.

        Args:
            player: Current player
            remaining_dice: Dice values not yet used
            board: Current board state
            current_sequence: Moves made so far
            all_sequences: List to accumulate all valid sequences
        """
        if not remaining_dice:
            # No more dice to use
            all_sequences.append(MoveSequence(current_sequence.copy()))
            return

        # Try using each remaining die
        found_move = False
        for i, die_value in enumerate(remaining_dice):
            possible_moves = self._get_possible_moves_for_die(player, die_value, board)

            for move in possible_moves:
                # Apply the move
                new_board = board.copy()
                new_board.apply_move(player, move.from_point, move.to_point)

                # Recurse with remaining dice
                new_remaining = remaining_dice[:i] + remaining_dice[i + 1:]
                new_sequence = current_sequence + [move]
                self._generate_move_sequences(player, new_remaining, new_board, new_sequence, all_sequences)
                found_move = True

        # If no move found with any remaining die, this is a terminal sequence
        if not found_move and current_sequence:
            all_sequences.append(MoveSequence(current_sequence.copy()))

    def _get_possible_moves_for_die(self, player: int, die_value: int, board: Board) -> List[Move]:
        """Get all possible moves for a single die value.

        Args:
            player: 1 or -1
            die_value: Die value to use
            board: Current board state

        Returns:
            List of possible moves
        """
        moves = []

        # First, check if we need to enter from bar
        if board.bar[player] > 0:
            # Must enter from bar
            if player == 1:
                entry_point = die_value
            else:
                entry_point = 25 - die_value

            if board.can_enter_from_bar(player, entry_point):
                moves.append(Move(0, entry_point, die_value))
            return moves  # Can only move from bar when pieces are there

        # Check bearing off
        can_bear_off = board.can_bear_off(player)

        # Check all points where player has pieces
        for point in range(1, 25):
            if board.get_piece_count(point, player) == 0:
                continue

            # Calculate destination
            if player == 1:
                to_point = point + die_value
            else:
                to_point = point - die_value

            # Check bearing off
            if can_bear_off:
                if player == 1:
                    if to_point >= 25:
                        # Exact or overshoot bearing off
                        if to_point == 25 or (to_point > 25 and self._is_highest_point(board, player, point)):
                            moves.append(Move(point, 25, die_value))
                            continue
                else:
                    if to_point <= 0:
                        # Exact or overshoot bearing off
                        if to_point == 0 or (to_point < 0 and self._is_highest_point(board, player, point)):
                            moves.append(Move(point, 0, die_value))
                            continue

            # Regular move
            if 1 <= to_point <= 24:
                # Check if destination is valid
                opponent = -player
                dest_value = board.points[to_point]

                # Can't move to opponent's made point (2+ pieces)
                if (opponent == 1 and dest_value >= 2) or (opponent == -1 and dest_value <= -2):
                    continue

                moves.append(Move(point, to_point, die_value))

        return moves

    def _is_highest_point(self, board: Board, player: int, point: int) -> bool:
        """Check if this is the highest point with player's pieces (for bearing off with overshoot).

        Args:
            board: Board state
            player: 1 or -1
            point: Point to check

        Returns:
            True if no pieces on higher points
        """
        if player == 1:
            for p in range(point + 1, 7):
                if board.get_piece_count(p, player) > 0:
                    return False
        else:
            for p in range(point - 1, 18, -1):
                if board.get_piece_count(p, player) > 0:
                    return False
        return True

    def _board_hash(self, board: Board) -> int:
        """Create a hash of the board state for duplicate detection.

        Args:
            board: Board to hash

        Returns:
            Hash value
        """
        return hash((tuple(board.points), tuple(board.bar.items()), tuple(board.off.items())))

    def apply_move_sequence(self, player: int, move_sequence: MoveSequence):
        """Apply a move sequence to the game board.

        Args:
            player: Player making the move
            move_sequence: Sequence of moves to apply
        """
        for move in move_sequence.moves:
            self.board.apply_move(player, move.from_point, move.to_point)

    def switch_player(self):
        """Switch to the other player."""
        self.current_player = -self.current_player

    def is_game_over(self) -> Optional[int]:
        """Check if game is over.

        Returns:
            Winner (1 or -1) if game over, None otherwise
        """
        return self.board.is_game_over()

    def play_move(self, player: int, move_sequence: MoveSequence):
        """Play a move and switch players.

        Args:
            player: Player making the move
            move_sequence: Sequence of moves to play
        """
        self.apply_move_sequence(player, move_sequence)
        self.switch_player()
