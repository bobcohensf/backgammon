"""
Flask web server for Backgammon game with web GUI.

This server provides a REST API for playing backgammon in a web browser.
It manages game state, handles move staging, and integrates with the AI player.
"""

import os
import sys
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import uuid

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.game import BackgammonGame, Move, MoveSequence
from ai.agent import TDAgent, MatchAwareTDAgent

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Store game sessions in memory (in production, use a database or Redis)
games = {}


def serialize_board(game):
    """Convert board state to JSON-serializable format."""
    return {
        'points': game.board.points[1:],  # Skip index 0 (unused)
        'bar': game.board.bar.copy(),
        'off': game.board.off.copy(),
        'current_player': game.current_player,
        'dice': game.dice if hasattr(game, 'dice') and game.dice else [],
        'winner': game.is_game_over(),
        'cube_value': game.board.cube_value,
        'cube_owner': game.board.cube_owner,
        'double_offered_by': game.board.double_offered_by
    }


def serialize_move(move):
    """Convert a Move object to JSON-serializable format."""
    return {
        'from': move.from_point,
        'to': move.to_point,
        'die': move.die_value
    }


def serialize_move_sequence(move_seq):
    """Convert a MoveSequence to JSON-serializable format."""
    return {
        'moves': [serialize_move(m) for m in move_seq.moves],
        'dice_used': move_seq.dice_used
    }


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('web', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Initialize a new game session or match."""
    data = request.json or {}
    match_length = data.get('match_length', 7)  # Default to 7 point match

    game_id = str(uuid.uuid4())

    # Initialize game and AI agent
    game = BackgammonGame()

    # Try to load match-aware model first, fall back to regular model
    match_model_path = 'models/match_model_final.pth'
    regular_model_path = 'models/model_final.pth'

    agent = None
    agent_type = 'regular'

    if os.path.exists(match_model_path):
        try:
            agent = MatchAwareTDAgent(epsilon=0.0)
            agent.match_length = match_length
            agent.load(match_model_path)
            agent_type = 'match_aware'
            print(f"Loaded match-aware model from {match_model_path}")
        except Exception as e:
            print(f"Warning: Could not load match model from {match_model_path}: {e}")

    if agent is None and os.path.exists(regular_model_path):
        try:
            agent = TDAgent(epsilon=0.0)
            agent.load(regular_model_path)
            agent_type = 'regular'
            print(f"Loaded regular model from {regular_model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {regular_model_path}: {e}")

    if agent is None:
        # No model found, create new regular agent
        agent = TDAgent(epsilon=0.0)
        agent_type = 'regular'
        print("No trained model found, using untrained agent")

    # Store game state
    games[game_id] = {
        'game': game,
        'agent': agent,
        'agent_type': agent_type,  # 'regular' or 'match_aware'
        'staged_moves': [],  # Moves staged but not yet accepted
        'original_board': None,  # Backup for reset
        'dice_rolled': False,
        # Match state
        'match_length': match_length,
        'match_score': {1: 0, -1: 0},
        'games_won': {1: 0, -1: 0},
        'crawford_game': False,  # Whether we're in Crawford game
        'post_crawford': False,  # Whether we're post-Crawford
        'game_history': []  # History of game results
    }

    return jsonify({
        'game_id': game_id,
        'board': serialize_board(game),
        'match_score': {1: 0, -1: 0},
        'match_length': match_length,
        'message': f'New match to {match_length} points started! Roll dice to begin.'
    })


@app.route('/api/game_state/<game_id>', methods=['GET'])
def get_game_state(game_id):
    """Get current game state."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    return jsonify({
        'board': serialize_board(game),
        'staged_moves': session['staged_moves'],
        'dice_rolled': session['dice_rolled'],
        'match_score': session['match_score'],
        'match_length': session['match_length'],
        'crawford_game': session['crawford_game']
    })


@app.route('/api/roll_dice/<game_id>', methods=['POST'])
def roll_dice(game_id):
    """Roll dice for the current player."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    # Check if game is over
    winner = game.is_game_over()
    if winner is not None:
        return jsonify({'error': 'Game is over', 'winner': winner}), 400

    # Roll dice
    dice = game.roll_dice()
    session['dice_rolled'] = True
    session['staged_moves'] = []
    # Store original dice as list (roll_dice returns tuple, but game.dice is a list)
    session['original_dice'] = game.dice[:]  # Copy game.dice which is always a list

    # Create backup of current board for reset functionality
    session['original_board'] = {
        'points': game.board.points[:],
        'bar': game.board.bar.copy(),
        'off': game.board.off.copy()
    }

    # Debug logging
    print(f"\n=== ROLL DICE DEBUG ===")
    print(f"Current player: {game.current_player}")
    print(f"Dice rolled: {dice}")
    print(f"game.dice: {game.dice}")

    # Check if there are any legal moves
    legal_moves = game.get_legal_moves(game.current_player, game.dice)
    print(f"Legal move sequences: {len(legal_moves)}")
    if legal_moves:
        print(f"First few legal moves:")
        for i, move_seq in enumerate(legal_moves[:5]):
            if move_seq.moves:
                moves_str = " -> ".join([f"{m.from_point}→{m.to_point}" for m in move_seq.moves])
                print(f"  {i}: {moves_str}")
    print("=" * 40 + "\n")

    return jsonify({
        'dice': dice,
        'board': serialize_board(game),
        'has_legal_moves': len(legal_moves) > 0,
        'num_legal_moves': len(legal_moves)
    })


@app.route('/api/legal_moves/<game_id>', methods=['GET'])
def get_legal_moves(game_id):
    """Get all legal move sequences for current dice."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    if not session['dice_rolled']:
        return jsonify({'error': 'Must roll dice first'}), 400

    legal_moves = game.get_legal_moves(game.current_player, game.dice)

    return jsonify({
        'legal_moves': [serialize_move_sequence(ms) for ms in legal_moves],
        'count': len(legal_moves)
    })


@app.route('/api/stage_move/<game_id>', methods=['POST'])
def stage_move(game_id):
    """Stage a single move (apply it temporarily for visual feedback)."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    data = request.json
    from_point = data.get('from')
    to_point = data.get('to')

    if from_point is None or to_point is None:
        return jsonify({'error': 'Must provide from and to points'}), 400

    session = games[game_id]
    game = session['game']

    if not session['dice_rolled']:
        return jsonify({'error': 'Must roll dice first'}), 400

    # Validate this move is legal
    # Note: legal_moves are calculated from the CURRENT board state (after previous staged moves)
    # so we always look at the first move in each sequence (index 0)
    legal_moves = game.get_legal_moves(game.current_player, game.dice)

    # Find if any legal move sequence starts with this move
    valid = False
    die_used = None

    for move_seq in legal_moves:
        if len(move_seq.moves) > 0:
            # Always check the first move since legal_moves are from current state
            next_move = move_seq.moves[0]
            if next_move.from_point == from_point and next_move.to_point == to_point:
                valid = True
                die_used = next_move.die_value
                break

    if not valid:
        return jsonify({'error': 'Invalid move'}), 400

    # Apply the move to the board
    try:
        game.board.apply_move(game.current_player, from_point, to_point)

        # Record the staged move
        session['staged_moves'].append({
            'from': from_point,
            'to': to_point,
            'die': die_used
        })

        # Remove the used die from the dice list
        if die_used in game.dice:
            game.dice.remove(die_used)

        return jsonify({
            'board': serialize_board(game),
            'staged_moves': session['staged_moves'],
            'remaining_dice': game.dice
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/reset_moves/<game_id>', methods=['POST'])
def reset_moves(game_id):
    """Reset all staged moves and restore original board state."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    if session['original_board'] is None:
        return jsonify({'error': 'No moves to reset'}), 400

    # Restore original board state
    game.board.points = session['original_board']['points'][:]
    game.board.bar = session['original_board']['bar'].copy()
    game.board.off = session['original_board']['off'].copy()

    # Re-roll to restore dice (or use stored dice)
    if not session.get('original_dice'):
        # Store the original dice on first roll
        session['original_dice'] = game.dice[:]

    game.dice = session['original_dice'][:]  # Restore as list

    # Clear staged moves
    session['staged_moves'] = []

    return jsonify({
        'board': serialize_board(game),
        'message': 'Moves reset'
    })


@app.route('/api/accept_turn/<game_id>', methods=['POST'])
def accept_turn(game_id):
    """Accept the current turn and switch to the next player."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    if not session['dice_rolled']:
        return jsonify({'error': 'Must roll dice first'}), 400

    # Check if game is over
    winner = game.is_game_over()
    if winner is not None:
        # Calculate points for this game
        points = game.board.get_points_for_win(winner)

        # Update match score
        session['match_score'][winner] += points
        session['games_won'][winner] += 1

        # Record game result
        game_result = {
            'winner': winner,
            'points': points,
            'cube_value': game.board.cube_value,
            'is_gammon': game.board.is_gammon(winner),
            'is_backgammon': game.board.is_backgammon(winner)
        }
        session['game_history'].append(game_result)

        # Check if match is over
        match_winner = None
        if session['match_score'][1] >= session['match_length']:
            match_winner = 1
        elif session['match_score'][-1] >= session['match_length']:
            match_winner = -1

        if match_winner is not None:
            return jsonify({
                'board': serialize_board(game),
                'game_over': True,
                'match_over': True,
                'winner': winner,
                'match_winner': match_winner,
                'match_score': session['match_score'],
                'points_scored': points,
                'game_type': 'backgammon' if game_result['is_backgammon'] else ('gammon' if game_result['is_gammon'] else 'normal'),
                'game_history': session['game_history']
            })

        # Match continues - check for Crawford rule
        crawford_status = ''
        if not session['crawford_game'] and not session['post_crawford']:
            # Check if anyone is one point away from winning
            for player in [1, -1]:
                if session['match_score'][player] == session['match_length'] - 1:
                    session['crawford_game'] = True
                    crawford_status = ' (Crawford Game - no doubling cube)'
                    break
        elif session['crawford_game']:
            # End Crawford game, enter post-Crawford
            session['crawford_game'] = False
            session['post_crawford'] = True

        # Start new game in the match
        game.board = game.board.__class__()  # Reset board
        game.current_player = 1  # Player 1 always starts
        game.dice = []

        # Reset turn state
        session['dice_rolled'] = False
        session['staged_moves'] = []
        session['original_board'] = None
        session['original_dice'] = None

        return jsonify({
            'board': serialize_board(game),
            'game_over': True,
            'match_over': False,
            'winner': winner,
            'match_score': session['match_score'],
            'match_length': session['match_length'],
            'points_scored': points,
            'game_type': 'backgammon' if game_result['is_backgammon'] else ('gammon' if game_result['is_gammon'] else 'normal'),
            'crawford_game': session['crawford_game'],
            'message': f"Game won by Player {winner}! {points} points scored.{crawford_status} Match score: {session['match_score'][1]}-{session['match_score'][-1]}. Click 'Roll Dice' to start next game."
        })

    # Switch to next player
    game.switch_player()

    # Reset turn state
    session['dice_rolled'] = False
    session['staged_moves'] = []
    session['original_board'] = None
    session['original_dice'] = None
    game.dice = []

    return jsonify({
        'board': serialize_board(game),
        'message': 'Turn accepted',
        'game_over': False,
        'match_score': session['match_score']
    })


@app.route('/api/ai_move/<game_id>', methods=['POST'])
def ai_move(game_id):
    """Execute AI move for computer player."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']
    agent = session['agent']
    agent_type = session.get('agent_type', 'regular')

    if not session['dice_rolled']:
        return jsonify({'error': 'Must roll dice first'}), 400

    # Get AI's move - use match context if match-aware agent
    if agent_type == 'match_aware':
        player = game.current_player
        my_score = session['match_score'][player]
        opp_score = session['match_score'][-player]
        is_crawford = session['crawford_game']

        best_move = agent.select_move(
            game, player, game.dice, my_score, opp_score, is_crawford, greedy=True
        )
    else:
        best_move = agent.select_move(game, game.current_player, game.dice, greedy=True)

    if best_move is None or (len(best_move.moves) == 0):
        return jsonify({
            'board': serialize_board(game),
            'message': 'No legal moves available',
            'moves': []
        })

    # Apply the move
    game.apply_move_sequence(game.current_player, best_move)

    # Record staged moves for display
    session['staged_moves'] = [serialize_move(m) for m in best_move.moves]

    return jsonify({
        'board': serialize_board(game),
        'moves': session['staged_moves'],
        'message': 'AI move executed'
    })


@app.route('/api/get_valid_destinations/<game_id>/<int:from_point>', methods=['GET'])
def get_valid_destinations(game_id, from_point):
    """Get valid destination points for a piece at the given position."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    if not session['dice_rolled']:
        return jsonify({'destinations': []})

    # Debug logging
    print(f"\n=== GET VALID DESTINATIONS DEBUG ===")
    print(f"From point: {from_point}")
    print(f"Current player: {game.current_player}")
    print(f"Dice: {game.dice}")
    print(f"Board point {from_point}: {game.board.points[from_point] if 1 <= from_point <= 24 else 'N/A'}")
    print(f"Bar: {game.board.bar}")

    # Get all legal moves from current board state
    legal_moves = game.get_legal_moves(game.current_player, game.dice)
    print(f"Total legal move sequences: {len(legal_moves)}")

    # Find destinations for pieces at from_point
    destinations = set()

    for i, move_seq in enumerate(legal_moves):
        # Check if this move sequence starts with a move from from_point
        # Always check first move (index 0) since legal_moves are from current state
        if len(move_seq.moves) > 0:
            next_move = move_seq.moves[0]
            print(f"  Sequence {i}: first move from {next_move.from_point} to {next_move.to_point}")
            if next_move.from_point == from_point:
                destinations.add(next_move.to_point)
                print(f"    -> MATCH! Added destination {next_move.to_point}")

    print(f"Valid destinations: {destinations}")
    print("=" * 40 + "\n")

    return jsonify({
        'destinations': list(destinations)
    })


@app.route('/api/offer_double/<game_id>', methods=['POST'])
def offer_double(game_id):
    """Offer a double to the opponent."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    # Check if in Crawford game
    if session['crawford_game']:
        return jsonify({'error': 'Cannot double during Crawford game'}), 400

    # Check if dice have been rolled
    if session['dice_rolled']:
        return jsonify({'error': 'Cannot double after rolling dice'}), 400

    try:
        game.board.offer_double(game.current_player)

        return jsonify({
            'board': serialize_board(game),
            'message': f'Player {game.current_player} offers a double to {game.board.cube_value * 2}!'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/accept_double/<game_id>', methods=['POST'])
def accept_double(game_id):
    """Accept a double offer."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    try:
        game.board.accept_double(game.current_player)

        return jsonify({
            'board': serialize_board(game),
            'message': f'Player {game.current_player} accepts! Cube is now {game.board.cube_value}. Roll dice to continue.'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/ai_cube_decision/<game_id>', methods=['POST'])
def ai_cube_decision(game_id):
    """AI decides whether to accept or reject a double.

    This is called automatically when AI is offered a double.
    """
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']
    agent = session['agent']
    agent_type = session.get('agent_type', 'regular')

    if game.board.double_offered_by is None:
        return jsonify({'error': 'No double has been offered'}), 400

    # The AI is the opponent of whoever offered the double
    # double_offered_by is the player who offered (e.g., player 1)
    # AI needs to be the opponent (e.g., player -1)
    opponent_player = -game.board.double_offered_by

    should_accept = True  # Default behavior

    if agent_type == 'match_aware':
        # Use AI's cube decision logic
        my_score = session['match_score'][opponent_player]
        opp_score = session['match_score'][-opponent_player]
        is_crawford = session['crawford_game']

        should_accept = agent.should_accept_double(
            game.board, opponent_player, my_score, opp_score, is_crawford
        )
    else:
        # Regular agent - use simple heuristic (always accept for now)
        # Could be enhanced with position evaluation
        should_accept = True

    if should_accept:
        # Accept the double
        game.board.accept_double(opponent_player)
        return jsonify({
            'decision': 'accept',
            'board': serialize_board(game),
            'message': f'AI accepts! Cube is now {game.board.cube_value}.'
        })
    else:
        # Reject the double - handled in reject_double endpoint
        return jsonify({
            'decision': 'reject',
            'message': 'AI rejects the double and forfeits the game.'
        })


@app.route('/api/reject_double/<game_id>', methods=['POST'])
def reject_double(game_id):
    """Reject a double offer (forfeit the game)."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']

    if game.board.double_offered_by is None:
        return jsonify({'error': 'No double has been offered'}), 400

    # The player who offered the double wins
    winner = game.board.double_offered_by
    points = game.board.cube_value  # Win current cube value (not doubled)

    game.board.reject_double()

    # Update match score
    session['match_score'][winner] += points
    session['games_won'][winner] += 1

    # Record game result
    game_result = {
        'winner': winner,
        'points': points,
        'cube_value': game.board.cube_value,
        'is_gammon': False,
        'is_backgammon': False,
        'double_rejected': True
    }
    session['game_history'].append(game_result)

    # Check if match is over
    match_winner = None
    if session['match_score'][1] >= session['match_length']:
        match_winner = 1
    elif session['match_score'][-1] >= session['match_length']:
        match_winner = -1

    if match_winner is not None:
        return jsonify({
            'board': serialize_board(game),
            'game_over': True,
            'match_over': True,
            'winner': winner,
            'match_winner': match_winner,
            'match_score': session['match_score'],
            'points_scored': points,
            'double_rejected': True,
            'message': f'Double rejected! Player {winner} wins {points} point(s). Match over!'
        })

    # Match continues - check for Crawford rule
    crawford_status = ''
    if not session['crawford_game'] and not session['post_crawford']:
        for player in [1, -1]:
            if session['match_score'][player] == session['match_length'] - 1:
                session['crawford_game'] = True
                crawford_status = ' (Crawford Game - no doubling cube)'
                break
    elif session['crawford_game']:
        session['crawford_game'] = False
        session['post_crawford'] = True

    # Start new game in the match
    game.board = game.board.__class__()  # Reset board
    game.current_player = 1
    game.dice = []

    # Reset turn state
    session['dice_rolled'] = False
    session['staged_moves'] = []
    session['original_board'] = None
    session['original_dice'] = None

    return jsonify({
        'board': serialize_board(game),
        'game_over': True,
        'match_over': False,
        'winner': winner,
        'match_score': session['match_score'],
        'points_scored': points,
        'double_rejected': True,
        'crawford_game': session['crawford_game'],
        'message': f"Double rejected! Player {winner} wins {points} point(s).{crawford_status} Match score: {session['match_score'][1]}-{session['match_score'][-1]}. Click 'Roll Dice' to start next game."
    })


@app.route('/api/match_stats/<game_id>', methods=['GET'])
def get_match_stats(game_id):
    """Get match statistics."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]

    return jsonify({
        'match_score': session['match_score'],
        'match_length': session['match_length'],
        'games_won': session['games_won'],
        'crawford_game': session['crawford_game'],
        'post_crawford': session['post_crawford'],
        'game_history': session['game_history']
    })


if __name__ == '__main__':
    print("Starting Backgammon Web Server...")
    print("Open your browser to http://localhost:5005")
    app.run(debug=True, host='0.0.0.0', port=5005)
