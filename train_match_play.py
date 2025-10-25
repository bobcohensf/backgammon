#!/usr/bin/env python3
"""
Training script for match-play backgammon AI with cube decisions.

This script trains the AI to optimize for match winning and make
proper cube decisions based on match score and position.

Usage:
    python train_match_play.py [--matches NUM] [--lr RATE] [--epsilon EPS]
"""

import argparse
import torch
from ai import MatchAwareTDAgent, MatchAwareBackgammonNet
from training import MatchPlayTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train backgammon AI for match play with cube decisions'
    )
    parser.add_argument('--matches', type=int, default=5000,
                       help='Number of matches to play (default: 5000)')
    parser.add_argument('--match-length', type=int, default=7,
                       help='Points to win each match (default: 7)')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--epsilon', type=float, default=0.15,
                       help='Exploration rate (default: 0.15)')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=0.7,
                       help='TD lambda parameter (default: 0.7)')
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N matches (default: 500)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load model from checkpoint path')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    parser.add_argument('--hidden-sizes', type=int, nargs='+',
                       default=[256, 128, 64],
                       help='Hidden layer sizes (default: 256 128 64)')

    args = parser.parse_args()

    print("=" * 70)
    print("Backgammon Match Play AI Training with Cube Decisions")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  Matches: {args.matches}")
    print(f"  Match Length: {args.match_length} points")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epsilon (exploration): {args.epsilon}")
    print(f"  Lambda (TD): {args.lambda_param}")
    print(f"  Hidden Layers: {args.hidden_sizes}")

    # Create or load network
    network = MatchAwareBackgammonNet(hidden_sizes=args.hidden_sizes)
    if args.load:
        print(f"\nLoading model from {args.load}...")
        network.load(args.load)

    # Create agent
    agent = MatchAwareTDAgent(
        network=network,
        learning_rate=args.lr,
        lambda_param=args.lambda_param,
        epsilon=args.epsilon
    )

    # Create trainer
    trainer = MatchPlayTrainer(
        agent,
        match_length=args.match_length,
        save_dir=args.save_dir
    )

    # Train
    print(f"\nStarting training...\n")
    trainer.train(
        num_matches=args.matches,
        save_every=args.save_every,
        verbose=True
    )

    # Evaluate final model
    print("\n" + "=" * 70)
    print("Evaluating final model...")
    print("=" * 70)
    results = trainer.evaluate(num_matches=100)
    print(f"\nEvaluation results (100 matches to {args.match_length}):")
    print(f"  Player 1 match wins: {results['p1_match_wins']} "
          f"({results['p1_match_win_rate']:.1%})")
    print(f"  Player 2 match wins: {results['p2_match_wins']} "
          f"({results['p2_match_win_rate']:.1%})")

    print(f"\n" + "=" * 70)
    print(f"Training complete!")
    print(f"Final model saved to {args.save_dir}/match_model_final.pth")
    print(f"\nTo use this model in the web interface:")
    print(f"  1. Copy to models/model_final.pth")
    print(f"  2. Update web_server.py to use MatchAwareTDAgent")
    print("=" * 70)


if __name__ == '__main__':
    main()
