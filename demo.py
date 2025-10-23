#!/usr/bin/env python3
"""
Quick demo to verify installation and show basic functionality.
"""

from game import BackgammonGame, Board
from ai import TDAgent, BackgammonNet


def demo_game_engine():
    """Demonstrate the game engine."""
    print("=" * 60)
    print("DEMO: Backgammon Game Engine")
    print("=" * 60)

    # Create a game
    game = BackgammonGame()

    print("\nInitial board position:")
    print(game.board)

    # Roll dice
    die1, die2 = game.roll_dice()
    print(f"\nPlayer X rolled: {die1}, {die2}")

    # Get legal moves
    legal_moves = game.get_legal_moves(1, game.dice)
    print(f"\nFound {len(legal_moves)} legal move sequences")

    if legal_moves:
        print("\nFirst few legal moves:")
        for i, move_seq in enumerate(legal_moves[:3]):
            print(f"{i+1}. ", end="")
            if move_seq.moves:
                for move in move_seq.moves:
                    print(f"{move.from_point}->{move.to_point}", end=" ")
            else:
                print("Pass", end="")
            print()

    print("\n✓ Game engine working correctly!")


def demo_neural_network():
    """Demonstrate the neural network."""
    print("\n" + "=" * 60)
    print("DEMO: Neural Network")
    print("=" * 60)

    # Create network
    network = BackgammonNet()
    print(f"\nCreated neural network:")
    print(f"  Input size: 196 features")
    print(f"  Hidden layers: 256 → 128 → 64")
    print(f"  Output: Win probability (0-1)")

    # Test evaluation
    board = Board()
    prob = network.evaluate(board, player=1)
    print(f"\nEvaluation of starting position for Player 1: {prob:.4f}")
    print("(Should be close to 0.5 for untrained network)")

    print("\n✓ Neural network working correctly!")


def demo_agent():
    """Demonstrate the AI agent."""
    print("\n" + "=" * 60)
    print("DEMO: TD Agent")
    print("=" * 60)

    # Create agent
    agent = TDAgent(epsilon=0.0)
    print("\nCreated TD agent with:")
    print("  Learning rate: 0.001")
    print("  Lambda: 0.7")
    print("  Epsilon: 0.0 (greedy)")

    # Play a few moves
    game = BackgammonGame()
    print("\nAgent playing first 3 turns:")

    for turn in range(3):
        player = game.current_player
        game.roll_dice()
        symbol = "X" if player == 1 else "O"

        print(f"\nTurn {turn + 1}: Player {symbol} rolls {game.dice}")

        move_seq = agent.select_move(game, player, game.dice, greedy=True)

        if move_seq.moves:
            print(f"Agent moves: ", end="")
            for i, move in enumerate(move_seq.moves):
                print(f"{move.from_point}->{move.to_point}", end="")
                if i < len(move_seq.moves) - 1:
                    print(", ", end="")
            print()
        else:
            print("Agent passes (no legal moves)")

        game.apply_move_sequence(player, move_seq)
        game.switch_player()

        if game.is_game_over():
            break

    print("\n✓ AI agent working correctly!")


def main():
    print("\n" + "=" * 60)
    print("Backgammon AI - Installation Verification")
    print("=" * 60)
    print("\nThis demo verifies that all components are working correctly.\n")

    try:
        demo_game_engine()
        demo_neural_network()
        demo_agent()

        print("\n" + "=" * 60)
        print("SUCCESS! All components working correctly.")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Train the AI:  python train.py --games 10000")
        print("  2. Play vs AI:    python play.py")
        print("  3. Evaluate:      python evaluate.py --model1 models/model_final.pth --random")
        print()

    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Something went wrong!")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPlease check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise


if __name__ == '__main__':
    main()
