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
from ai.agent import TDAgent

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
        'winner': game.is_game_over()
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
    """Initialize a new game session."""
    game_id = str(uuid.uuid4())

    # Initialize game and AI agent
    game = BackgammonGame()
    agent = TDAgent(epsilon=0.0)  # AI plays optimally (no exploration)

    # Try to load trained model if it exists
    model_path = 'models/model_final.pth'
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")

    # Store game state
    games[game_id] = {
        'game': game,
        'agent': agent,
        'staged_moves': [],  # Moves staged but not yet accepted
        'original_board': None,  # Backup for reset
        'dice_rolled': False
    }

    return jsonify({
        'game_id': game_id,
        'board': serialize_board(game),
        'message': 'New game created. Roll dice to start!'
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
        'dice_rolled': session['dice_rolled']
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

    # Create backup of current board for reset functionality
    session['original_board'] = {
        'points': game.board.points[:],
        'bar': game.board.bar.copy(),
        'off': game.board.off.copy()
    }

    # Check if there are any legal moves
    legal_moves = game.get_legal_moves(game.current_player, game.dice)

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

    game.dice = session['original_dice'][:]

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
        return jsonify({
            'board': serialize_board(game),
            'game_over': True,
            'winner': winner
        })

    # Switch to next player
    game.switch_player()

    # Reset turn state
    session['dice_rolled'] = False
    session['staged_moves'] = []
    session['original_board'] = None
    session['original_dice'] = None
    game.dice = []

    # Check if game is over after this turn
    winner = game.is_game_over()

    return jsonify({
        'board': serialize_board(game),
        'message': 'Turn accepted',
        'game_over': winner is not None,
        'winner': winner
    })


@app.route('/api/ai_move/<game_id>', methods=['POST'])
def ai_move(game_id):
    """Execute AI move for computer player."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404

    session = games[game_id]
    game = session['game']
    agent = session['agent']

    if not session['dice_rolled']:
        return jsonify({'error': 'Must roll dice first'}), 400

    # Get AI's move
    legal_moves = game.get_legal_moves(game.current_player, game.dice)

    if not legal_moves:
        return jsonify({
            'board': serialize_board(game),
            'message': 'No legal moves available',
            'moves': []
        })

    # AI selects best move
    best_move = agent.select_move(game, game.current_player, game.dice, greedy=True)

    if best_move is None:
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

    # Get all legal moves from current board state
    legal_moves = game.get_legal_moves(game.current_player, game.dice)

    # Find destinations for pieces at from_point
    destinations = set()

    for move_seq in legal_moves:
        # Check if this move sequence starts with a move from from_point
        # Always check first move (index 0) since legal_moves are from current state
        if len(move_seq.moves) > 0:
            next_move = move_seq.moves[0]
            if next_move.from_point == from_point:
                destinations.add(next_move.to_point)

    return jsonify({
        'destinations': list(destinations)
    })


if __name__ == '__main__':
    print("Starting Backgammon Web Server...")
    print("Open your browser to http://localhost:5005")
    app.run(debug=True, host='0.0.0.0', port=5005)
