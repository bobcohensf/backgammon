#!/usr/bin/env python3
"""
Main training script for backgammon AI.

Usage:
    python train.py [--games NUM] [--lr RATE] [--epsilon EPS]
"""

import argparse
import torch
from ai import TDAgent, BackgammonNet
from training import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description='Train backgammon AI using TD learning')
    parser.add_argument('--games', type=int, default=50000,
                       help='Number of games to play (default: 50000)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Exploration rate (default: 0.1)')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=0.7,
                       help='TD lambda parameter (default: 0.7)')
    parser.add_argument('--save-every', type=int, default=5000,
                       help='Save checkpoint every N games (default: 5000)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load model from checkpoint path')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models (default: models)')

    args = parser.parse_args()

    print("=" * 60)
    print("Backgammon AI Training with TD(λ) Learning")
    print("=" * 60)

    # Create or load agent
    network = BackgammonNet()
    if args.load:
        print(f"Loading model from {args.load}...")
        network.load(args.load)

    agent = TDAgent(
        network=network,
        learning_rate=args.lr,
        lambda_param=args.lambda_param,
        epsilon=args.epsilon
    )

    # Create trainer
    trainer = SelfPlayTrainer(agent, save_dir=args.save_dir)

    # Train
    trainer.train(
        num_games=args.games,
        save_every=args.save_every,
        verbose=True
    )

    # Evaluate final model
    print("\n" + "=" * 60)
    print("Evaluating final model...")
    print("=" * 60)
    results = trainer.evaluate(num_games=1000)
    print(f"Evaluation results (1000 games):")
    print(f"  Player 1 wins: {results['p1_wins']} ({results['p1_win_rate']:.1%})")
    print(f"  Player 2 wins: {results['p2_wins']} ({results['p2_win_rate']:.1%})")

    print(f"\nTraining complete! Final model saved to {args.save_dir}/model_final.pth")


if __name__ == '__main__':
    main()
