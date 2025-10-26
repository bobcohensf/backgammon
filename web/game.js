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
let matchScore = {1: 0, '-1': 0};
let matchLength = 7;
let crawfordGame = false;

// DOM elements
const newGameBtn = document.getElementById('new-game-btn');
const rollDiceBtn = document.getElementById('roll-dice-btn');
const doubleBtn = document.getElementById('double-btn');
const resetMoveBtn = document.getElementById('reset-move-btn');
const acceptMoveBtn = document.getElementById('accept-move-btn');
const acceptDoubleBtn = document.getElementById('accept-double-btn');
const rejectDoubleBtn = document.getElementById('reject-double-btn');
const currentPlayerDisplay = document.getElementById('current-player');
const diceDisplay = document.getElementById('dice-display');
const cubeDisplay = document.getElementById('cube-display');
const cubeOwner = document.getElementById('cube-owner');
const statusMessage = document.getElementById('status-message');
const doubleOfferPanel = document.getElementById('double-offer-panel');
const doubleOfferText = document.getElementById('double-offer-text');
const player1Score = document.getElementById('player1-score');
const player2Score = document.getElementById('player2-score');
const matchTarget = document.getElementById('match-target');
const crawfordIndicator = document.getElementById('crawford-indicator');
const gamesWon = document.getElementById('games-won');
const historyList = document.getElementById('history-list');

// Event listeners
newGameBtn.addEventListener('click', startNewGame);
rollDiceBtn.addEventListener('click', rollDice);
doubleBtn.addEventListener('click', offerDouble);
resetMoveBtn.addEventListener('click', resetMoves);
acceptMoveBtn.addEventListener('click', acceptTurn);
acceptDoubleBtn.addEventListener('click', acceptDouble);
rejectDoubleBtn.addEventListener('click', rejectDouble);

/**
 * Start a new game
 */
async function startNewGame() {
    try {
        updateStatus('Starting new match...');

        const response = await fetch(`${API_BASE}/api/new_game`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ match_length: 7 })
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        gameId = data.game_id;
        matchScore = data.match_score;
        matchLength = data.match_length;
        crawfordGame = false;

        renderBoard(data.board);
        updateMatchScore();
        updateStatus(data.message);

        // Enable roll dice button for player 1
        rollDiceBtn.disabled = false;
        resetMoveBtn.disabled = true;
        acceptMoveBtn.disabled = true;
        diceRolled = false;
        doubleOfferPanel.style.display = 'none';

        // Update double button based on initial game state
        updateDoubleButton(data.board);
        updateCubeDisplay(data.board);

        // Clear history
        historyList.innerHTML = 'No games played yet';

        console.log('New match started:', gameId);
    } catch (error) {
        console.error('Error starting new match:', error);
        updateStatus('Failed to start new match', 'error');
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
        updateCubeDisplay(data.board);

        rollDiceBtn.disabled = true;
        doubleBtn.disabled = true;
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
            // Update match score
            matchScore = data.match_score;
            updateMatchScore();

            // Update game history
            if (data.game_history) {
                updateGameHistory(data.game_history);
            }

            if (data.match_over) {
                const matchWinner = data.match_winner === 1 ? 'Player 1 (X)' : 'Player 2 (O)';
                updateStatus(`MATCH OVER! ${matchWinner} wins the match!`);
                rollDiceBtn.disabled = true;
                doubleBtn.disabled = true;
                resetMoveBtn.disabled = true;
                acceptMoveBtn.disabled = true;
                return;
            }

            // Game over but match continues
            const winner = data.winner === 1 ? 'Player 1' : 'Player 2';
            updateStatus(data.message || `Game won by ${winner}! Click "Roll Dice" to start next game.`);

            // Update Crawford indicator
            crawfordGame = data.crawford_game || false;
            crawfordIndicator.style.display = crawfordGame ? 'block' : 'none';

            rollDiceBtn.disabled = false;
            resetMoveBtn.disabled = true;
            acceptMoveBtn.disabled = true;
            diceRolled = false;

            // Update double button based on game state (Crawford, cube ownership, etc.)
            updateDoubleButton(data.board);
            updateCubeDisplay(data.board);

            return;
        }

        // Switch player
        currentPlayer = data.board.current_player;
        diceRolled = false;
        selectedPoint = null;
        validDestinations = [];

        // Update match score if present
        if (data.match_score) {
            matchScore = data.match_score;
            updateMatchScore();
        }

        rollDiceBtn.disabled = false;
        resetMoveBtn.disabled = true;
        acceptMoveBtn.disabled = true;
        updateDoubleButton(data.board);

        updateDiceDisplay([]);
        updateCubeDisplay(data.board);
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

    await selectPoint(0); // 0 represents the bar
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
        updateCubeDisplay(boardData);
        updateDoubleButton(boardData);
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
            const checker = createChecker(1, 0);
            barPlayer1.appendChild(checker);
        }
        if (boardData.bar[1] > 3) {
            barPlayer1.lastChild.textContent = boardData.bar[1];
        }
    }

    if (boardData.bar[-1] > 0) {
        for (let i = 0; i < Math.min(boardData.bar[-1], 3); i++) {
            const checker = createChecker(-1, 0);
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

    // Clear OFF area highlights
    document.getElementById('off-area-top').classList.remove('highlight');
    document.getElementById('off-area-bottom').classList.remove('highlight');

    // Highlight selected point
    if (selectedPoint !== null && selectedPoint !== 0) {
        const selectedElement = document.querySelector(`[data-point="${selectedPoint}"]`);
        if (selectedElement) {
            selectedElement.classList.add('selected');
        }
    }

    // Highlight valid destinations
    validDestinations.forEach(dest => {
        if (dest === 25) {
            // Player 1 bearing off - highlight top OFF area
            document.getElementById('off-area-top').classList.add('highlight');
        } else if (dest === 0) {
            // Player 2 bearing off - highlight bottom OFF area
            document.getElementById('off-area-bottom').classList.add('highlight');
        } else {
            // Regular point destination
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

    // Add click handlers to OFF areas for bearing off
    document.getElementById('off-area-top').onclick = () => {
        if (validDestinations.includes(25)) {
            stageMove(selectedPoint, 25);
        }
    };
    document.getElementById('off-area-bottom').onclick = () => {
        if (validDestinations.includes(0)) {
            stageMove(selectedPoint, 0);
        }
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
 * Offer a double to the opponent
 */
async function offerDouble() {
    if (!gameId || diceRolled) return;

    try {
        const response = await fetch(`${API_BASE}/api/offer_double/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);
        updateCubeDisplay(data.board);

        // Show double offer panel to opponent
        if (currentPlayer === 1) {
            // Human offered, AI needs to respond (auto-respond for now)
            updateStatus('You offered a double! AI is thinking...');
            setTimeout(() => aiRespondToDouble(), 500);
        } else {
            // AI offered, show panel to human
            showDoubleOffer();
        }

    } catch (error) {
        console.error('Error offering double:', error);
        updateStatus('Failed to offer double', 'error');
    }
}

/**
 * Accept a double offer
 */
async function acceptDouble() {
    if (!gameId) return;

    try {
        const response = await fetch(`${API_BASE}/api/accept_double/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);
        updateCubeDisplay(data.board);
        doubleOfferPanel.style.display = 'none';

        updateStatus(data.message);
        rollDiceBtn.disabled = false;
        doubleBtn.disabled = true;

    } catch (error) {
        console.error('Error accepting double:', error);
        updateStatus('Failed to accept double', 'error');
    }
}

/**
 * Reject a double offer (forfeit game)
 */
async function rejectDouble() {
    if (!gameId) return;

    try {
        const response = await fetch(`${API_BASE}/api/reject_double/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        renderBoard(data.board);
        doubleOfferPanel.style.display = 'none';

        // Update match score
        matchScore = data.match_score;
        updateMatchScore();

        // Update Crawford indicator
        crawfordGame = data.crawford_game || false;
        crawfordIndicator.style.display = crawfordGame ? 'block' : 'none';

        if (data.match_over) {
            const matchWinner = data.match_winner === 1 ? 'Player 1 (X)' : 'Player 2 (O)';
            updateStatus(`MATCH OVER! ${matchWinner} wins the match!`);
            rollDiceBtn.disabled = true;
            doubleBtn.disabled = true;
        } else {
            updateStatus(data.message);
            rollDiceBtn.disabled = false;
            doubleBtn.disabled = true;
        }

    } catch (error) {
        console.error('Error rejecting double:', error);
        updateStatus('Failed to reject double', 'error');
    }
}

/**
 * Show double offer panel
 */
function showDoubleOffer() {
    doubleOfferText.textContent = 'Opponent offers to double!';
    doubleOfferPanel.style.display = 'block';
    rollDiceBtn.disabled = true;
    doubleBtn.disabled = true;
}

/**
 * AI responds to double using trained cube decision logic
 */
async function aiRespondToDouble() {
    if (!gameId) return;

    try {
        const response = await fetch(`${API_BASE}/api/ai_cube_decision/${gameId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            return;
        }

        if (data.decision === 'accept') {
            // AI accepted - cube is now owned by AI
            renderBoard(data.board);
            updateCubeDisplay(data.board);
            updateStatus(data.message);
            rollDiceBtn.disabled = false;
            // Update double button based on new cube ownership
            updateDoubleButton(data.board);
        } else {
            // AI rejected - game forfeited, handled in reject endpoint
            await rejectDouble();
        }

    } catch (error) {
        console.error('Error with AI cube decision:', error);
        updateStatus('Failed to process AI cube decision', 'error');
    }
}

/**
 * Update cube display
 */
function updateCubeDisplay(board) {
    if (!board) return;

    cubeDisplay.textContent = board.cube_value;

    let ownerText = 'Center';
    if (board.cube_owner === 1) {
        ownerText = 'Player 1';
    } else if (board.cube_owner === -1) {
        ownerText = 'Player 2 (AI)';
    }
    cubeOwner.textContent = ownerText;

    // Show double offer panel if a double is offered
    if (board.double_offered_by !== null && board.double_offered_by !== currentPlayer) {
        showDoubleOffer();
    }
}

/**
 * Update double button state
 */
function updateDoubleButton(board) {
    // Add stack trace to see who called this
    const stack = new Error().stack;
    console.log('[DEBUG] updateDoubleButton called:', {
        board: board ? 'present' : 'null',
        diceRolled,
        crawfordGame,
        currentPlayer,
        cube_owner: board?.cube_owner,
        cube_value: board?.cube_value,
        calledFrom: stack.split('\n')[2]?.trim()  // Show who called this function
    });

    if (!board || diceRolled || crawfordGame) {
        console.log('[DEBUG] Disabling double button: no board, dice rolled, or Crawford');
        doubleBtn.disabled = true;
        return;
    }

    // Enable double button if current player can double and it's player 1
    const canDouble = board.cube_owner === null || board.cube_owner === currentPlayer;
    const shouldEnable = currentPlayer === 1 && canDouble && board.cube_value < 64;

    console.log('[DEBUG] Double button logic:', {
        canDouble,
        shouldEnable,
        'cube_owner === null': board.cube_owner === null,
        'cube_owner === currentPlayer': board.cube_owner === currentPlayer,
        'currentPlayer === 1': currentPlayer === 1,
        'cube_value < 64': board.cube_value < 64
    });

    const newState = !shouldEnable;
    console.log('[DEBUG] Setting doubleBtn.disabled =', newState, '(was', doubleBtn.disabled, ')');
    doubleBtn.disabled = newState;
}

/**
 * Update match score display
 */
function updateMatchScore() {
    player1Score.textContent = `Player 1: ${matchScore[1] || 0}`;
    player2Score.textContent = `Player 2: ${matchScore[-1] || 0}`;
    matchTarget.textContent = `First to ${matchLength}`;
}

/**
 * Update game history
 */
function updateGameHistory(history) {
    if (!history || history.length === 0) {
        historyList.innerHTML = 'No games played yet';
        return;
    }

    let html = '<ul>';
    history.forEach((game, index) => {
        const winner = game.winner === 1 ? 'P1' : 'P2';
        const type = game.is_backgammon ? 'Backgammon' : (game.is_gammon ? 'Gammon' : 'Normal');
        const rejected = game.double_rejected ? ' (Double Rejected)' : '';
        html += `<li>Game ${index + 1}: ${winner} wins ${game.points}pt (${type}${rejected})</li>`;
    });
    html += '</ul>';
    historyList.innerHTML = html;

    // Update games won
    const p1Wins = history.filter(g => g.winner === 1).length;
    const p2Wins = history.filter(g => g.winner === -1).length;
    gamesWon.textContent = `Player 1: ${p1Wins} | Player 2: ${p2Wins}`;
}

/**
 * Initialize the game on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    updateStatus('Click "New Game" to start!');
    updatePlayerDisplay();
    updateMatchScore();
});
