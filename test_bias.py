#!/usr/bin/env python3
"""
Test if the game has any systematic bias toward Player 1 or Player 2.
Plays games with random moves to see if win rates are ~50/50.
"""

from game import BackgammonGame
import random


def play_random_game():
    """Play a game with completely random moves."""
    game = BackgammonGame()

    for _ in range(500):  # Max moves
        player = game.current_player
        game.roll_dice()

        legal_moves = game.get_legal_moves(player, game.dice)
        if legal_moves:
            move_seq = random.choice(legal_moves)
            game.apply_move_sequence(player, move_seq)

        winner = game.is_game_over()
        if winner:
            return winner

        game.switch_player()

    # Didn't finish - count by pieces off
    if game.board.off[1] > game.board.off[-1]:
        return 1
    elif game.board.off[-1] > game.board.off[1]:
        return -1
    else:
        return 1  # Tie goes to P1


def main():
    print("Testing for systematic bias with random play...")
    print("Playing 1000 games with purely random moves...\n")

    wins = {1: 0, -1: 0}

    for i in range(1000):
        winner = play_random_game()
        wins[winner] += 1

        if (i + 1) % 100 == 0:
            p1_rate = wins[1] / (i + 1)
            print(f"{i+1} games: P1={wins[1]} ({p1_rate:.1%}), P2={wins[-1]} ({(1-p1_rate):.1%})")

    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Player 1 wins: {wins[1]} ({wins[1]/1000:.1%})")
    print(f"Player 2 wins: {wins[-1]} ({wins[-1]/1000:.1%})")
    print("="*60)

    p1_rate = wins[1] / 1000
    if 0.48 <= p1_rate <= 0.52:
        print("✓ No significant bias detected")
    else:
        print(f"⚠️  BIAS DETECTED! Expected ~50%, got {p1_rate:.1%}")
        print("This suggests a bug in the game engine.")


if __name__ == '__main__':
    main()
