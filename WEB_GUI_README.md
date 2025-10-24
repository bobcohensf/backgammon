# Backgammon Web GUI

A beautiful web-based graphical interface for playing backgammon against an AI opponent powered by TD-Learning.

## Features

- **Interactive Board**: Click-based piece movement with visual feedback
- **AI Opponent**: Play against a trained TD-Learning neural network
- **Move Staging**: Experiment with different moves before committing
- **Visual Highlights**: See valid destinations for selected pieces
- **Reset Functionality**: Undo moves and try different strategies
- **Responsive Design**: Works on desktop and mobile browsers

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install Flask and Flask-CORS in addition to the existing dependencies.

## Running the Web Server

1. Start the Flask web server:

```bash
python web_server.py
```

2. Open your web browser and navigate to:

```
http://localhost:5005
```

3. The server will automatically load the trained AI model from `models/model_final.pth` if available.

## How to Play

### Starting a Game

1. Click the **"New Game"** button to initialize a new game
2. Player 1 (X - white checkers) plays against Player 2 (O - black checkers, AI)

### Playing Your Turn

1. Click **"Roll Dice"** to roll the dice
2. The dice values will be displayed at the top
3. Click on one of your checkers to select it
4. Valid destination points will be highlighted in yellow
5. Click on a highlighted destination to move the checker
6. Repeat until you've used all your dice or can't make more moves
7. Click **"Reset Move"** at any time to undo all moves and start over
8. Click **"Accept Move"** when you're satisfied with your moves

### AI Turn

1. After you accept your move, it becomes the AI's turn
2. The game automatically rolls dice for the AI
3. The AI calculates and executes its best move
4. Click **"Accept Move"** to proceed to your next turn

### Game Rules

- **Player 1 (X)**: Moves UP from point 1 → 24, home board is points 19-24
- **Player 2 (O)**: Moves DOWN from point 24 → 1, home board is points 1-6
- **Hit pieces** go to the bar and must re-enter before other moves
- **Bearing off** is only allowed when all pieces are in the home board
- **Doubles** (same number on both dice) give you 4 moves instead of 2

### Winning

The first player to bear off all 15 checkers wins the game!

## UI Components

### Board Layout

```
 Points 13-18  |  BAR  |  Points 19-24  |  OFF
----------------------------------------------------
                  △△△△△△
 Points 12-7   |  BAR  |  Points 6-1    |  OFF
```

- **Points**: The 24 triangular positions where checkers can be placed
- **Bar**: Center area where hit checkers are placed
- **Off**: Area showing how many checkers each player has borne off

### Controls

- **New Game**: Start a fresh game
- **Roll Dice**: Roll the dice for the current player
- **Reset Move**: Undo all staged moves for the current turn
- **Accept Move**: Confirm your moves and end your turn

### Visual Indicators

- **Green glow**: Selected piece
- **Yellow glow**: Valid destination points
- **Pulsing animation**: Clickable pieces during your turn

## Technical Details

### Backend (web_server.py)

- **Framework**: Flask with CORS support
- **Game State**: Managed in-memory (session-based)
- **AI Integration**: Uses TDAgent with trained neural network
- **Move Validation**: Ensures all moves follow backgammon rules

### Frontend (web/)

- **index.html**: Game board structure and UI layout
- **styles.css**: Beautiful backgammon board styling with gradient effects
- **game.js**: Client-side game logic and API communication

### API Endpoints

- `POST /api/new_game`: Initialize a new game session
- `GET /api/game_state/<game_id>`: Get current game state
- `POST /api/roll_dice/<game_id>`: Roll dice for current player
- `GET /api/legal_moves/<game_id>`: Get all legal move sequences
- `POST /api/stage_move/<game_id>`: Stage a single move
- `POST /api/reset_moves/<game_id>`: Reset all staged moves
- `POST /api/accept_turn/<game_id>`: Accept turn and switch players
- `POST /api/ai_move/<game_id>`: Execute AI move
- `GET /api/get_valid_destinations/<game_id>/<point>`: Get valid destinations for a piece

## Troubleshooting

### Server won't start

- Make sure Flask is installed: `pip install flask flask-cors`
- Check that port 5000 is not already in use
- Try running on a different port: modify the last line of `web_server.py`

### AI not loading

- Train a model first using: `python train.py --games 10000`
- Or the server will work without a trained model (AI will make random moves)

### Board not displaying correctly

- Make sure you're using a modern browser (Chrome, Firefox, Safari, Edge)
- Try clearing your browser cache
- Check the browser console for JavaScript errors

## Next Steps

- Train a better AI model for stronger gameplay
- Implement save/load game functionality
- Add multiplayer support
- Track game statistics and player ratings
- Implement doubling cube for advanced play

## Credits

Built on top of the TD-Gammon inspired backgammon engine with PyTorch neural networks.
