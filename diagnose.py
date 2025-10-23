#!/usr/bin/env python3
"""
Diagnostic script to see what's happening in games.
"""

from game import BackgammonGame
from ai import TDAgent, BackgammonNet
import os


def watch_game(agent, max_moves=100):
    """Watch a game and print what's happening."""
    game = BackgammonGame()

    print("Starting diagnostic game...")
    print(game.board)

    for move_num in range(max_moves):
        player = game.current_player
        symbol = "X" if player == 1 else "O"

        # Roll dice
        game.roll_dice()
        print(f"\nMove {move_num + 1}: {symbol} rolled {game.dice}")

        # Get move
        move_seq = agent.select_move(game, player, game.dice, greedy=True)

        if not move_seq.moves:
            print(f"  {symbol} cannot move")
        else:
            print(f"  {symbol} moves: ", end="")
            for i, move in enumerate(move_seq.moves):
                from_str = "BAR" if move.from_point == 0 else str(move.from_point)
                to_str = "OFF" if move.to_point in [0, 25] else str(move.to_point)
                print(f"{from_str}->{to_str}", end="")
                if i < len(move_seq.moves) - 1:
                    print(", ", end="")
            print()

        # Apply move
        game.apply_move_sequence(player, move_seq)

        # Check game state every 10 moves
        if (move_num + 1) % 10 == 0:
            print(f"\n--- After {move_num + 1} moves ---")
            print(game.board)
            print(f"P1 pieces off: {game.board.off[1]}, P2 pieces off: {game.board.off[-1]}")

        # Check for winner
        winner = game.is_game_over()
        if winner is not None:
            print(f"\n{'='*60}")
            print(f"Game over after {move_num + 1} moves!")
            print(f"Winner: {'X' if winner == 1 else 'O'}")
            print(game.board)
            return winner

        game.switch_player()

    print(f"\n{'='*60}")
    print(f"Game did not finish after {max_moves} moves")
    print("Final board state:")
    print(game.board)
    print(f"P1 pieces off: {game.board.off[1]}, P2 pieces off: {game.board.off[-1]}")
    return None


def main():
    print("Loading model...")
    network = BackgammonNet()

    # Try to load trained model
    if os.path.exists('models/model_final.pth'):
        network.load('models/model_final.pth')
        print("Loaded trained model")
    else:
        print("Using untrained model")

    agent = TDAgent(network=network, epsilon=0.0)

    print("\n" + "="*60)
    print("Running diagnostic game (max 100 moves)")
    print("="*60)

    winner = watch_game(agent, max_moves=100)

    if winner is None:
        print("\n⚠️  Game did not finish - this is the problem!")
        print("The AI is not making progress toward winning.")
    else:
        print("\n✓ Game finished successfully!")


if __name__ == '__main__':
    main()
