"""Diagnose training issues with doubling and win rate."""

import torch
import numpy as np
from game import BackgammonGame
from ai.agent import MatchAwareTDAgent
from ai.network import MatchAwareBackgammonNet

# Load the trained model
model_path = "models/match_model_final.pth"
print("Loading model from:", model_path)

network = MatchAwareBackgammonNet()
network.load_state_dict(torch.load(model_path))
network.eval()

agent = MatchAwareTDAgent(network, epsilon=0.0)
agent.match_length = 7

# Test 1: Check cube decision outputs for various positions
print("\n" + "="*70)
print("TEST 1: Cube Decision Output Values")
print("="*70)

game = BackgammonGame()
print("\nStarting position cube decisions:")

for player in [1, -1]:
    for my_score in [0, 3, 6]:
        opp_score = 3
        cube_decision = agent.network.evaluate_cube_decision(
            game.board, player, my_score, opp_score, 7, False
        )
        print(f"  Player {player:2d}, Score {my_score}-{opp_score}: "
              f"should_double={cube_decision['should_double']:.4f}, "
              f"should_accept={cube_decision['should_accept']:.4f}")

# Test 2: Check match equity evaluation symmetry
print("\n" + "="*70)
print("TEST 2: Match Equity Symmetry Check")
print("="*70)

game = BackgammonGame()
my_score, opp_score = 3, 3

# Evaluate from both perspectives
equity_p1 = agent.network.evaluate_match_equity(
    game.board, 1, my_score, opp_score, 7, False
)
equity_p2 = agent.network.evaluate_match_equity(
    game.board, -1, opp_score, my_score, 7, False
)

print(f"\nStarting position, score 3-3:")
print(f"  Player 1 match equity: {equity_p1:.4f}")
print(f"  Player 2 match equity: {equity_p2:.4f}")
print(f"  Sum (should be ~1.0): {equity_p1 + equity_p2:.4f}")

if abs(equity_p1 - 0.5) > 0.1:
    print(f"\n⚠️  WARNING: Starting position shows bias toward Player 1!")
    print(f"  Expected ~0.5, got {equity_p1:.4f}")

# Test 3: Check if network learns to double by sampling positions
print("\n" + "="*70)
print("TEST 3: Doubling Threshold Analysis")
print("="*70)

threshold = 0.7
print(f"\nCurrent doubling threshold: {threshold}")
print("Checking 100 random mid-game positions...")

double_count = 0
should_double_values = []

for _ in range(100):
    # Create a random mid-game position
    game = BackgammonGame()
    for _ in range(20):  # Play 20 random moves
        game.roll_dice()
        legal_moves = game.get_legal_moves(game.current_player, game.dice)
        if legal_moves and legal_moves[0].moves:
            game.apply_move_sequence(game.current_player, legal_moves[0])
            if game.is_game_over():
                break
            game.switch_player()

    if not game.is_game_over():
        cube_decision = agent.network.evaluate_cube_decision(
            game.board, game.current_player, 3, 3, 7, False
        )
        should_double_values.append(cube_decision['should_double'])
        if cube_decision['should_double'] > threshold:
            double_count += 1

if should_double_values:
    print(f"\nPositions that would trigger double: {double_count}/100")
    print(f"should_double values:")
    print(f"  Min:  {min(should_double_values):.4f}")
    print(f"  Max:  {max(should_double_values):.4f}")
    print(f"  Mean: {np.mean(should_double_values):.4f}")
    print(f"  Std:  {np.std(should_double_values):.4f}")

    if max(should_double_values) < threshold:
        print(f"\n⚠️  PROBLEM: Max should_double ({max(should_double_values):.4f}) "
              f"never exceeds threshold ({threshold})!")
        print(f"  Recommended: Lower threshold to ~{max(should_double_values) * 0.8:.2f}")

# Test 4: Check position evaluation for both players
print("\n" + "="*70)
print("TEST 4: Position Evaluation Symmetry")
print("="*70)

print("\nPlaying one game with move evaluations...")
game = BackgammonGame()
move_evals = {1: [], -1: []}

for move_num in range(50):
    if game.is_game_over():
        break

    player = game.current_player
    game.roll_dice()

    # Get best move
    legal_moves = game.get_legal_moves(player, game.dice)
    if not legal_moves or not legal_moves[0].moves:
        game.switch_player()
        continue

    # Evaluate position before move
    equity = agent.network.evaluate_match_equity(game.board, player, 3, 3, 7, False)
    move_evals[player].append(equity)

    game.apply_move_sequence(player, legal_moves[0])
    game.switch_player()

print(f"\nPlayer 1 average equity: {np.mean(move_evals[1]):.4f} "
      f"(n={len(move_evals[1])})")
print(f"Player 2 average equity: {np.mean(move_evals[-1]):.4f} "
      f"(n={len(move_evals[-1])})")

if abs(np.mean(move_evals[1]) - 0.5) > 0.15:
    print(f"\n⚠️  PROBLEM: Significant bias detected!")
    print(f"  In self-play, both players should average ~0.5 equity")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
