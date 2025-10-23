#!/usr/bin/env python3
"""
Evaluate and benchmark backgammon AI models.

Usage:
    python evaluate.py --model1 PATH [--model2 PATH] [--games NUM]
"""

import argparse
from ai import TDAgent, BackgammonNet
from training import SelfPlayTrainer


class RandomAgent:
    """Baseline random agent for comparison."""

    def __init__(self):
        self.epsilon = 1.0  # Always random

    def select_move(self, game, player, dice, greedy=False):
        """Select random legal move."""
        from game import MoveSequence
        import random

        legal_moves = game.get_legal_moves(player, dice)
        if not legal_moves:
            return MoveSequence([])
        return random.choice(legal_moves)


def main():
    parser = argparse.ArgumentParser(description='Evaluate backgammon AI models')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model')
    parser.add_argument('--model2', type=str, default=None,
                       help='Path to second model (optional, uses random if not specified)')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of evaluation games (default: 1000)')
    parser.add_argument('--random', action='store_true',
                       help='Evaluate against random baseline')

    args = parser.parse_args()

    print("=" * 60)
    print("Backgammon AI Evaluation")
    print("=" * 60)

    # Load first model
    print(f"\nLoading model 1 from {args.model1}...")
    network1 = BackgammonNet()
    try:
        network1.load(args.model1)
        agent1 = TDAgent(network=network1, epsilon=0.0)
        print("Model 1 loaded successfully")
    except FileNotFoundError:
        print(f"Error: Model file {args.model1} not found!")
        return

    # Load second model or use random
    if args.random or args.model2 is None:
        print("Using random agent as opponent")
        agent2 = RandomAgent()
        opponent_name = "Random"
    else:
        print(f"\nLoading model 2 from {args.model2}...")
        network2 = BackgammonNet()
        try:
            network2.load(args.model2)
            agent2 = TDAgent(network=network2, epsilon=0.0)
            print("Model 2 loaded successfully")
            opponent_name = "Model 2"
        except FileNotFoundError:
            print(f"Error: Model file {args.model2} not found!")
            return

    # Evaluate
    print(f"\n{'=' * 60}")
    print(f"Running {args.games} games...")
    print(f"{'=' * 60}")

    trainer = SelfPlayTrainer(agent1)

    # First half: Model 1 as player 1
    print(f"\nPhase 1: Model 1 as X (first player)")
    results1 = trainer.evaluate(num_games=args.games // 2, opponent=agent2)

    # Second half: Model 1 as player 2
    print(f"\nPhase 2: Model 1 as O (second player)")
    # Swap agents by creating new trainer with agent2
    if not isinstance(agent2, RandomAgent):
        trainer2 = SelfPlayTrainer(agent2)
        results2 = trainer2.evaluate(num_games=args.games // 2, opponent=agent1)
        # Reverse perspective for combined results
        model1_wins_as_p2 = results2['p2_wins']
        model1_losses_as_p2 = results2['p1_wins']
    else:
        # For random agent, we can use the same approach
        # But we need a custom evaluation
        from game import BackgammonGame
        from tqdm import tqdm

        wins = 0
        losses = 0
        for _ in tqdm(range(args.games // 2), desc="Evaluating"):
            game = BackgammonGame()
            move_count = 0
            max_moves = 500

            while move_count < max_moves:
                player = game.current_player
                agent = agent2 if player == 1 else agent1  # Model 1 as player 2

                game.roll_dice()
                move_seq = agent.select_move(game, player, game.dice, greedy=True)
                game.apply_move_sequence(player, move_seq)

                winner = game.is_game_over()
                if winner is not None:
                    if winner == -1:  # Model 1 wins
                        wins += 1
                    else:
                        losses += 1
                    break

                game.switch_player()
                move_count += 1

        model1_wins_as_p2 = wins
        model1_losses_as_p2 = losses

    # Combined results
    total_wins = results1['p1_wins'] + model1_wins_as_p2
    total_games = args.games
    win_rate = total_wins / total_games

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Total games: {total_games}")
    print(f"\nModel 1 performance:")
    print(f"  As X (first):  {results1['p1_wins']}/{args.games // 2} "
          f"({results1['p1_win_rate']:.1%})")
    print(f"  As O (second): {model1_wins_as_p2}/{args.games // 2} "
          f"({model1_wins_as_p2 / (args.games // 2):.1%})")
    print(f"\nOverall: {total_wins}/{total_games} ({win_rate:.1%})")
    print(f"{'=' * 60}")

    # Interpretation
    if win_rate > 0.55:
        print("✓ Model 1 is significantly stronger!")
    elif win_rate > 0.50:
        print("✓ Model 1 has a slight advantage")
    elif win_rate > 0.45:
        print("≈ Models are approximately equal")
    else:
        print(f"✗ {opponent_name} is stronger")


if __name__ == '__main__':
    main()
