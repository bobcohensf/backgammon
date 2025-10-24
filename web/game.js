/**
 * Backgammon Web Game Client
 * Handles all game logic, UI updates, and server communication
 */

// API configuration
const API_BASE = window.location.origin;

// Game state
let gameId = null;
let currentPlayer = 1;
let dice = [];
let selectedPoint = null;
let validDestinations = [];
let isPlayerTurn = true;
let diceRolled = false;

// DOM elements
const newGameBtn = document.getElementById('new-game-btn');
const rollDiceBtn = document.getElementById('roll-dice-btn');
const resetMoveBtn = document.getElementById('reset-move-btn');
const acceptMoveBtn = document.getElementById('accept-move-btn');
const currentPlayerDisplay = document.getElementById('current-player');
const diceDisplay = document.getElementById('dice-display');
const statusMessage = document.getElementById('status-message');

// Event listeners
newGameBtn.addEventListener('click', startNewGame);
rollDiceBtn.addEventListener('click', rollDice);
resetMoveBtn.addEventListener('click', resetMoves);
acceptMoveBtn.addEventListener('click', acceptTurn);

/**
 * Start a new game
 */
async function startNewGame() {
    try {
        updateStatus('Starting new game...');

        const response = await fetch(`${API_BASE}/api/new_game`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        gameId = data.game_id;
        renderBoard(data.board);
        updateStatus(data.message);

        // Enable roll dice button for player 1
        rollDiceBtn.disabled = false;
        resetMoveBtn.disabled = true;
        acceptMoveBtn.disabled = true;
        diceRolled = false;

        console.log('New game started:', gameId);
    } catch (error) {
        console.error('Error starting new game:', error);
        updateStatus('Failed to start new game', 'error');
    }
}

/**
 * Roll dice for current player
 */
async function rollDice() {
    if (!gameId) {
        updateStatus('Please start a new game first', 'error');
        return;
    }

    try {
        updateStatus('Rolling dice...');

        const response = await fetch(`${API_BASE}/api/roll_dice/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        dice = data.dice;
        diceRolled = true;
        renderBoard(data.board);
        updateDiceDisplay(dice);

        rollDiceBtn.disabled = true;
        resetMoveBtn.disabled = false;

        if (!data.has_legal_moves) {
            updateStatus('No legal moves available. Click "Accept Move" to end turn.');
            acceptMoveBtn.disabled = false;
        } else {
            // Check if it's AI's turn
            if (currentPlayer === -1) {
                updateStatus('AI is thinking...');
                setTimeout(() => executeAIMove(), 500);
            } else {
                updateStatus(`You rolled ${dice.join(', ')}. Click on a checker to move.`);
                acceptMoveBtn.disabled = true;
            }
        }

    } catch (error) {
        console.error('Error rolling dice:', error);
        updateStatus('Failed to roll dice', 'error');
    }
}

/**
 * Execute AI move
 */
async function executeAIMove() {
    if (!gameId) return;

    try {
        const response = await fetch(`${API_BASE}/api/ai_move/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);

        if (data.moves && data.moves.length > 0) {
            const moveDesc = data.moves.map(m => `${m.from}→${m.to}`).join(', ');
            updateStatus(`AI moved: ${moveDesc}. Click "Accept Move" to continue.`);
        } else {
            updateStatus('AI has no legal moves. Click "Accept Move" to continue.');
        }

        acceptMoveBtn.disabled = false;
        resetMoveBtn.disabled = true;

    } catch (error) {
        console.error('Error executing AI move:', error);
        updateStatus('Failed to execute AI move', 'error');
    }
}

/**
 * Reset staged moves
 */
async function resetMoves() {
    if (!gameId) return;

    try {
        updateStatus('Resetting moves...');

        const response = await fetch(`${API_BASE}/api/reset_moves/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);
        selectedPoint = null;
        validDestinations = [];
        updateStatus('Moves reset. Click on a checker to move.');
        acceptMoveBtn.disabled = true;

    } catch (error) {
        console.error('Error resetting moves:', error);
        updateStatus('Failed to reset moves', 'error');
    }
}

/**
 * Accept turn and switch players
 */
async function acceptTurn() {
    if (!gameId) return;

    try {
        updateStatus('Accepting turn...');

        const response = await fetch(`${API_BASE}/api/accept_turn/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);

        if (data.game_over) {
            const winner = data.winner === 1 ? 'Player 1 (X)' : 'Player 2 (O)';
            updateStatus(`Game Over! ${winner} wins!`);
            rollDiceBtn.disabled = true;
            resetMoveBtn.disabled = true;
            acceptMoveBtn.disabled = true;
            return;
        }

        // Switch player
        currentPlayer = data.board.current_player;
        diceRolled = false;
        selectedPoint = null;
        validDestinations = [];

        rollDiceBtn.disabled = false;
        resetMoveBtn.disabled = true;
        acceptMoveBtn.disabled = true;

        updateDiceDisplay([]);
        updateStatus('Turn accepted. Click "Roll Dice" to continue.');

        // Auto-roll for AI
        if (currentPlayer === -1) {
            updateStatus('AI turn. Rolling dice...');
            setTimeout(() => rollDice(), 1000);
        }

    } catch (error) {
        console.error('Error accepting turn:', error);
        updateStatus('Failed to accept turn', 'error');
    }
}

/**
 * Handle point click
 */
async function handlePointClick(point) {
    if (!gameId || !diceRolled || currentPlayer !== 1) {
        return;
    }

    // If a point is already selected, try to move to this point
    if (selectedPoint !== null) {
        if (validDestinations.includes(point)) {
            await stageMove(selectedPoint, point);
        } else {
            // Deselect current and select new point
            await selectPoint(point);
        }
    } else {
        await selectPoint(point);
    }
}

/**
 * Handle bar click
 */
async function handleBarClick() {
    if (!gameId || !diceRolled || currentPlayer !== 1) {
        return;
    }

    await selectPoint(25); // 25 represents the bar
}

/**
 * Select a point and get valid destinations
 */
async function selectPoint(point) {
    try {
        // Check if there's a piece of the current player at this point
        const response = await fetch(`${API_BASE}/api/get_valid_destinations/${gameId}/${point}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error getting valid destinations:', data.error);
            return;
        }

        if (data.destinations.length === 0) {
            updateStatus('No valid moves from this position.');
            selectedPoint = null;
            validDestinations = [];
            renderBoard(null); // Re-render to clear highlights
            return;
        }

        selectedPoint = point;
        validDestinations = data.destinations;
        renderBoard(null); // Re-render with highlights

        updateStatus(`Selected point ${point}. Click on a highlighted destination.`);

    } catch (error) {
        console.error('Error selecting point:', error);
    }
}

/**
 * Stage a move
 */
async function stageMove(from, to) {
    if (!gameId) return;

    try {
        const response = await fetch(`${API_BASE}/api/stage_move/${gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ from, to })
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Invalid move: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);
        selectedPoint = null;
        validDestinations = [];

        if (data.remaining_dice.length === 0) {
            updateStatus('All dice used. Click "Accept Move" to end turn.');
            acceptMoveBtn.disabled = false;
            resetMoveBtn.disabled = false;
        } else {
            updateStatus(`Move applied! ${data.remaining_dice.length} dice remaining. Continue moving or click "Accept Move".`);
            acceptMoveBtn.disabled = false;
            resetMoveBtn.disabled = false;
        }

    } catch (error) {
        console.error('Error staging move:', error);
        updateStatus('Failed to stage move', 'error');
    }
}

/**
 * Render the board
 */
function renderBoard(boardData) {
    if (boardData) {
        currentPlayer = boardData.current_player;
        updatePlayerDisplay();
    }

    // Get current board state from DOM if not provided
    if (!boardData) {
        // Just update highlights without changing pieces
        updateHighlights();
        return;
    }

    // Render points
    for (let i = 1; i <= 24; i++) {
        const pointElement = document.getElementById(`point-${i}`);
        pointElement.innerHTML = '';

        const count = boardData.points[i - 1];
        if (count !== 0) {
            const player = count > 0 ? 1 : -1;
            const absCount = Math.abs(count);

            for (let j = 0; j < Math.min(absCount, 5); j++) {
                const checker = createChecker(player, i);
                pointElement.appendChild(checker);
            }

            // Add count label if more than 5 checkers
            if (absCount > 5) {
                const lastChecker = pointElement.lastChild;
                lastChecker.textContent = absCount;
            }
        }
    }

    // Render bar
    const barPlayer1 = document.getElementById('bar-player-1');
    const barPlayer2 = document.getElementById('bar-player-2');

    barPlayer1.innerHTML = '';
    barPlayer2.innerHTML = '';

    if (boardData.bar[1] > 0) {
        for (let i = 0; i < Math.min(boardData.bar[1], 3); i++) {
            const checker = createChecker(1, 25);
            barPlayer1.appendChild(checker);
        }
        if (boardData.bar[1] > 3) {
            barPlayer1.lastChild.textContent = boardData.bar[1];
        }
    }

    if (boardData.bar[-1] > 0) {
        for (let i = 0; i < Math.min(boardData.bar[-1], 3); i++) {
            const checker = createChecker(-1, 25);
            barPlayer2.appendChild(checker);
        }
        if (boardData.bar[-1] > 3) {
            barPlayer2.lastChild.textContent = boardData.bar[-1];
        }
    }

    // Render off areas
    document.getElementById('off-player-1').textContent = boardData.off[1];
    document.getElementById('off-player-2').textContent = boardData.off[-1];

    // Update highlights
    updateHighlights();
}

/**
 * Create a checker element
 */
function createChecker(player, point) {
    const checker = document.createElement('div');
    checker.className = `checker player-${player === 1 ? '1' : '2'}`;

    if (player === currentPlayer && point === selectedPoint) {
        checker.classList.add('clickable');
    }

    return checker;
}

/**
 * Update highlights on the board
 */
function updateHighlights() {
    // Clear all highlights
    document.querySelectorAll('.point').forEach(point => {
        point.classList.remove('highlight', 'selected');
    });

    // Highlight selected point
    if (selectedPoint !== null && selectedPoint !== 25) {
        const selectedElement = document.querySelector(`[data-point="${selectedPoint}"]`);
        if (selectedElement) {
            selectedElement.classList.add('selected');
        }
    }

    // Highlight valid destinations
    validDestinations.forEach(dest => {
        if (dest !== 0) { // 0 is bearing off
            const destElement = document.querySelector(`[data-point="${dest}"]`);
            if (destElement) {
                destElement.classList.add('highlight');
            }
        }
    });

    // Add click handlers to all points
    document.querySelectorAll('.point').forEach(point => {
        const pointNum = parseInt(point.getAttribute('data-point'));
        point.onclick = () => handlePointClick(pointNum);
    });

    // Add click handlers to bar
    document.getElementById('bar-player-1').onclick = () => {
        if (currentPlayer === 1) handleBarClick();
    };
    document.getElementById('bar-player-2').onclick = () => {
        if (currentPlayer === -1) handleBarClick();
    };
}

/**
 * Update dice display
 */
function updateDiceDisplay(dice) {
    if (dice.length === 0) {
        diceDisplay.textContent = '-';
    } else {
        diceDisplay.textContent = dice.join(', ');
    }
}

/**
 * Update player display
 */
function updatePlayerDisplay() {
    const playerName = currentPlayer === 1 ? 'Player 1 (X)' : 'Player 2 (O)';
    currentPlayerDisplay.textContent = playerName;
    currentPlayerDisplay.style.color = currentPlayer === 1 ? '#007bff' : '#dc3545';
}

/**
 * Update status message
 */
function updateStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.style.background = type === 'error' ? '#ffe0e0' : '#e7f3ff';
    statusMessage.style.color = type === 'error' ? '#cc0000' : '#0056b3';
}

/**
 * Initialize the game on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    updateStatus('Click "New Game" to start!');
    updatePlayerDisplay();
});
