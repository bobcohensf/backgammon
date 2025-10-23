#!/usr/bin/env python3
"""
Detailed diagnostic for a normal game from starting position.
"""

from game import BackgammonGame


def detailed_game():
    """Run a game with detailed output."""
    game = BackgammonGame()

    print("Starting position:")
    print(game.board)
    print("\nPlayer 1 pieces at:")
    for p in range(1, 25):
        if game.board.points[p] > 0:
            print(f"  Point {p}: {game.board.points[p]} pieces")

    print("\nPlayer 2 pieces at:")
    for p in range(1, 25):
        if game.board.points[p] < 0:
            print(f"  Point {p}: {abs(game.board.points[p])} pieces")

    print("\n" + "="*60)
    print("Playing 20 turns with detailed output...")
    print("="*60)

    for turn in range(20):
        player = game.current_player
        symbol = "X" if player == 1 else "O"

        game.roll_dice()
        print(f"\nTurn {turn+1}: {symbol} rolled {game.dice}")

        legal_moves = game.get_legal_moves(player, game.dice)
        print(f"  Legal moves: {len(legal_moves)}")

        if legal_moves and legal_moves[0].moves:
            move_seq = legal_moves[0]
            print(f"  Applying: ", end="")
            for i, move in enumerate(move_seq.moves):
                from_str = "BAR" if move.from_point == 0 else str(move.from_point)
                to_str = "OFF" if move.to_point in [0, 25] else str(move.to_point)
                print(f"{from_str}→{to_str}", end="")
                if i < len(move_seq.moves) - 1:
                    print(", ", end="")
            print()

            game.apply_move_sequence(player, move_seq)
        else:
            print("  No legal moves!")

        # Show progress every 5 turns
        if (turn + 1) % 5 == 0:
            print(f"\n--- After {turn+1} turns ---")
            print(f"P1 on bar: {game.board.bar[1]}, P1 off: {game.board.off[1]}")
            print(f"P2 on bar: {game.board.bar[-1]}, P2 off: {game.board.off[-1]}")

            # Show where pieces are
            print("P1 pieces at points:", end="")
            for p in range(1, 25):
                if game.board.points[p] > 0:
                    print(f" {p}({game.board.points[p]})", end="")
            print()

            print("P2 pieces at points:", end="")
            for p in range(1, 25):
                if game.board.points[p] < 0:
                    print(f" {p}({abs(game.board.points[p])})", end="")
            print()

        winner = game.is_game_over()
        if winner:
            print(f"\n{'='*60}")
            print(f"Game over! Winner: {symbol}")
            print(game.board)
            return

        game.switch_player()

    print(f"\n{'='*60}")
    print("Game didn't finish in 20 turns")
    print(f"\nFinal state:")
    print(game.board)
    print(f"P1: on bar={game.board.bar[1]}, off={game.board.off[1]}")
    print(f"P2: on bar={game.board.bar[-1]}, off={game.board.off[-1]}")


if __name__ == '__main__':
    detailed_game()
