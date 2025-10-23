#!/usr/bin/env python3
"""
Interactive play against the trained AI.

Usage:
    python play.py [--model PATH]
"""

import argparse
from game import BackgammonGame, MoveSequence
from ai import TDAgent, BackgammonNet


class HumanPlayer:
    """Human player interface."""

    def select_move(self, game: BackgammonGame, player: int, dice: list) -> MoveSequence:
        """Let human select a move.

        Args:
            game: Current game
            player: Player number
            dice: Dice rolled

        Returns:
            Selected move sequence
        """
        legal_moves = game.get_legal_moves(player, dice)

        if not legal_moves or (len(legal_moves) == 1 and len(legal_moves[0].moves) == 0):
            print("No legal moves available!")
            return legal_moves[0] if legal_moves else MoveSequence([])

        print(f"\nYou have {len(legal_moves)} legal move(s):")
        for i, move_seq in enumerate(legal_moves):
            print(f"\n{i + 1}. ", end="")
            if not move_seq.moves:
                print("Pass (no moves)")
            else:
                for j, move in enumerate(move_seq.moves):
                    from_str = "BAR" if move.from_point == 0 else str(move.from_point)
                    to_str = "OFF" if move.to_point in [0, 25] else str(move.to_point)
                    print(f"{from_str}->{to_str}", end="")
                    if j < len(move_seq.moves) - 1:
                        print(", ", end="")
            print()

        while True:
            try:
                choice = input(f"\nSelect move (1-{len(legal_moves)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(legal_moves):
                    return legal_moves[idx]
                else:
                    print(f"Please enter a number between 1 and {len(legal_moves)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nGame interrupted!")
                exit(0)


def play_game(ai_agent: TDAgent, human_is_player1=True):
    """Play a game human vs AI.

    Args:
        ai_agent: Trained AI agent
        human_is_player1: If True, human plays as player 1 (X), else player 2 (O)
    """
    game = BackgammonGame()
    human = HumanPlayer()

    human_player = 1 if human_is_player1 else -1
    ai_player = -human_player

    human_symbol = "X" if human_player == 1 else "O"
    ai_symbol = "O" if human_player == 1 else "X"

    print("=" * 60)
    print("BACKGAMMON - Human vs AI")
    print("=" * 60)
    print(f"You are playing as: {human_symbol}")
    print(f"AI is playing as: {ai_symbol}")
    print("=" * 60)

    move_count = 0
    max_moves = 500

    while move_count < max_moves:
        current_player = game.current_player
        is_human = (current_player == human_player)

        # Show board
        print("\n" + game.board.__str__())

        # Roll dice
        die1, die2 = game.roll_dice()
        player_symbol = "X" if current_player == 1 else "O"

        print(f"\n{player_symbol}'s turn")
        print(f"Dice: {die1}, {die2}")

        if die1 == die2:
            print(f"Doubles! You get four {die1}'s")

        # Get move
        if is_human:
            move_seq = human.select_move(game, current_player, game.dice)
        else:
            print("\nAI is thinking...")
            move_seq = ai_agent.select_move(game, current_player, game.dice, greedy=True)

            # Show AI's move
            if not move_seq.moves:
                print("AI passes (no legal moves)")
            else:
                print("AI moves: ", end="")
                for i, move in enumerate(move_seq.moves):
                    from_str = "BAR" if move.from_point == 0 else str(move.from_point)
                    to_str = "OFF" if move.to_point in [0, 25] else str(move.to_point)
                    print(f"{from_str}->{to_str}", end="")
                    if i < len(move_seq.moves) - 1:
                        print(", ", end="")
                print()

        # Apply move
        game.apply_move_sequence(current_player, move_seq)

        # Check for winner
        winner = game.is_game_over()
        if winner is not None:
            print("\n" + "=" * 60)
            print(game.board.__str__())
            print("=" * 60)

            winner_symbol = "X" if winner == 1 else "O"
            if winner == human_player:
                print(f"🎉 Congratulations! You ({winner_symbol}) win!")
            else:
                print(f"AI ({winner_symbol}) wins! Better luck next time.")
            print("=" * 60)
            return winner

        game.switch_player()
        move_count += 1

    print("\nMaximum moves reached - game drawn")
    return None


def main():
    parser = argparse.ArgumentParser(description='Play backgammon against trained AI')
    parser.add_argument('--model', type=str, default='models/model_final.pth',
                       help='Path to trained model (default: models/model_final.pth)')
    parser.add_argument('--side', type=str, default='x', choices=['x', 'o'],
                       help='Play as X (first) or O (second) - default: x')

    args = parser.parse_args()

    # Load AI
    print("Loading AI model...")
    network = BackgammonNet()
    try:
        network.load(args.model)
        print(f"Model loaded from {args.model}")
    except FileNotFoundError:
        print(f"Warning: Model file {args.model} not found!")
        print("Using untrained model. Train first with: python train.py")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    agent = TDAgent(network=network, epsilon=0.0)  # No exploration during play

    human_is_player1 = (args.side == 'x')

    while True:
        play_game(agent, human_is_player1)

        response = input("\nPlay again? (y/n): ")
        if response.lower() != 'y':
            break

    print("\nThanks for playing!")


if __name__ == '__main__':
    main()
