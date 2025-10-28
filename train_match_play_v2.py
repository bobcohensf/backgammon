#!/usr/bin/env python3
"""
Improved training script for match-play backgammon AI with cube decisions.

Key improvements over v1:
- Lower initial doubling threshold (0.55 → 0.70 over training)
- Exploration bonus for cube decisions (0.15 → 0.00 over training)
- Alternating first player to measure first-move advantage
- Better tracking statistics
"""

import argparse
import os
import torch
from ai.network import MatchAwareBackgammonNet
from ai.agent import MatchAwareTDAgent
from training.match_trainer_v2 import ImprovedMatchPlayTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train match-aware backgammon AI with improved cube decision learning'
    )
    parser.add_argument('--matches', type=int, default=1000,
                       help='Number of matches to play (default: 1000)')
    parser.add_argument('--match-length', type=int, default=7,
                       help='Points to win match (default: 7)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Exploration rate for move selection (default: 0.1)')
    parser.add_argument('--lambda-param', type=float, default=0.7,
                       help='TD(λ) parameter (default: 0.7)')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden layer sizes (default: 256 128 64)')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save checkpoints (default: models)')
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N matches (default: 500)')
    parser.add_argument('--eval-matches', type=int, default=100,
                       help='Number of matches for evaluation (default: 100)')
    parser.add_argument('--double-threshold', type=float, default=None,
                       help='Fixed doubling threshold (overrides dynamic schedule)')
    parser.add_argument('--exploration-bonus', type=float, default=None,
                       help='Fixed exploration bonus (overrides dynamic schedule)')
    parser.add_argument('--continue-schedule', action='store_true',
                       help='Continue dynamic schedule from where last run left off')
    parser.add_argument('--max-cube', type=int, default=16,
                       help='Maximum cube value during training to prevent runaway (default: 16)')

    args = parser.parse_args()

    print("=" * 70)
    print("IMPROVED Backgammon Match Play AI Training with Cube Decisions")
    print("=" * 70)
    print("\nTraining Configuration:")
    print(f"  Matches: {args.matches}")
    print(f"  Match Length: {args.match_length} points")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epsilon (exploration): {args.epsilon}")
    print(f"  Lambda (TD): {args.lambda_param}")
    print(f"  Hidden Layers: {args.hidden}")
    print(f"  Max Cube Value: {args.max_cube if args.max_cube else 'unlimited'}")
    print()

    # Create or load network
    network = MatchAwareBackgammonNet(board_input_size=196, hidden_sizes=args.hidden)

    if args.load:
        print(f"Loading model from {args.load}...")
        network.load_state_dict(torch.load(args.load))
        print("Model loaded successfully!")
    else:
        print("Initializing new model...")

    # Create agent
    agent = MatchAwareTDAgent(
        network=network,
        learning_rate=args.lr,
        lambda_param=args.lambda_param,
        epsilon=args.epsilon
    )

    # Create improved trainer
    trainer = ImprovedMatchPlayTrainer(
        agent=agent,
        match_length=args.match_length,
        save_dir=args.save_dir,
        max_cube_value=args.max_cube
    )

    # Handle fixed vs dynamic hyperparameters
    if args.double_threshold is not None or args.exploration_bonus is not None:
        fixed_threshold = args.double_threshold if args.double_threshold is not None else 0.70
        fixed_exploration = args.exploration_bonus if args.exploration_bonus is not None else 0.00

        print("\nStarting training with FIXED hyperparameters...")
        print(f"  - Fixed doubling threshold: {fixed_threshold}")
        print(f"  - Fixed exploration bonus: {fixed_exploration}")
        print("  - Alternating first player each game")
        print("  - Tracking first-player advantage")
        print()

        # Train with fixed values
        trainer.train(num_matches=args.matches, save_every=args.save_every, verbose=True,
                     fixed_threshold=fixed_threshold, fixed_exploration=fixed_exploration)
    else:
        print("\nStarting training with DYNAMIC hyperparameters...")
        print("Key improvements:")
        print("  - Dynamic doubling threshold: 0.65 → 0.70")
        print("  - Exploration bonus for doubles: 0.10 → 0.00")
        print(f"  - Max cube value: {args.max_cube} (prevents runaway)")
        print("  - Alternating first player each game")
        print("  - Tracking first-player advantage")
        print()

        # Train with dynamic schedule
        trainer.train(num_matches=args.matches, save_every=args.save_every, verbose=True)

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating final model...")
    print("=" * 70)

    results = trainer.evaluate(num_matches=args.eval_matches)

    print(f"\nEvaluation results ({args.eval_matches} matches to {args.match_length}):")
    print(f"  Player 1 match wins: {results['p1_match_wins']} ({results['p1_match_win_rate']*100:.1f}%)")
    print(f"  Player 2 match wins: {results['p2_match_wins']} ({results['p2_match_win_rate']*100:.1f}%)")
    print(f"  First-player advantage: {results['first_player_advantage']*100:.1f}%")
    print(f"    (Expected: ~55-60% in backgammon)")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Final model saved to {args.save_dir}/match_model_final.pth")
    print("\nTo use this model in the web interface:")
    print("  1. Copy to models/model_final.pth")
    print("  2. Update web_server.py to use MatchAwareTDAgent")
    print("=" * 70)


if __name__ == '__main__':
    main()
