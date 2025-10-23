#!/usr/bin/env python3
"""
Test bearing off mechanics to ensure the game engine works correctly.
"""

from game import BackgammonGame, Board, Move, MoveSequence


def test_bearing_off():
    """Test that bearing off works correctly."""
    print("Testing bearing off mechanics...")

    # Create a board where player 1 can bear off
    board = Board([0] * 25)  # Empty board

    # Put Player 1's pieces in home board (points 19-24)
    board.points[19] = 3
    board.points[20] = 3
    board.points[21] = 3
    board.points[22] = 3
    board.points[23] = 3
    board.off[1] = 0

    # Put Player 2's pieces far away (negative for player 2)
    board.points[7] = -15
    board.off[-1] = 0

    print("\nInitial board:")
    print(board)

    game = BackgammonGame(board)
    game.current_player = 1

    print("\nPlayer 1 should be able to bear off...")
    print(f"Can bear off: {board.can_bear_off(1)}")

    # Simulate rolling and moving
    for turn in range(20):
        player = game.current_player
        game.roll_dice()

        if player == 1:
            print(f"\nTurn {turn + 1}: Player 1 rolled {game.dice}")
            legal_moves = game.get_legal_moves(1, game.dice)
            print(f"Legal moves: {len(legal_moves)}")

            if legal_moves and legal_moves[0].moves:
                move_seq = legal_moves[0]
                print(f"Applying: {move_seq}")
                game.apply_move_sequence(1, move_seq)
                print(f"Pieces off: {game.board.off[1]}")

                winner = game.is_game_over()
                if winner == 1:
                    print(f"\n✓ Player 1 won after {turn + 1} turns!")
                    print(f"Final pieces off: {game.board.off[1]}")
                    return True
        else:
            # Player 2 can't do anything, skip
            pass

        game.switch_player()

    print(f"\n✗ Player 1 didn't win after 20 turns")
    print(f"Pieces off: {game.board.off[1]}")
    return False


def test_basic_game():
    """Test a basic game from start."""
    print("\n" + "="*60)
    print("Testing basic game from start...")
    print("="*60)

    game = BackgammonGame()

    for turn in range(50):
        player = game.current_player
        game.roll_dice()

        legal_moves = game.get_legal_moves(player, game.dice)
        if legal_moves:
            game.apply_move_sequence(player, legal_moves[0])

        winner = game.is_game_over()
        if winner:
            print(f"\nGame ended after {turn + 1} turns!")
            print(f"Winner: {'Player 1' if winner == 1 else 'Player 2'}")
            print(f"P1 pieces off: {game.board.off[1]}")
            print(f"P2 pieces off: {game.board.off[-1]}")
            return True

        game.switch_player()

    print(f"\nGame didn't finish after 50 turns")
    print(f"P1 pieces off: {game.board.off[1]}")
    print(f"P2 pieces off: {game.board.off[-1]}")
    return False


if __name__ == '__main__':
    print("="*60)
    print("Backgammon Game Engine Tests")
    print("="*60)

    success1 = test_bearing_off()
    success2 = test_basic_game()

    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Bearing off test: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"  Basic game test: {'✓ PASS' if success2 else '✗ FAIL'}")
    print("="*60)
