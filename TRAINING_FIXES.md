# Training Issues and Fixes

## Issues Found in Your Training Run

### Issue 1: No Doubling Activity (0.00 doubles/game)

**Problem:**
The AI never offers doubles during training, which means the cube decision head of the neural network never receives training signals.

**Root Causes:**
1. **Threshold too high**: The doubling threshold is set to 0.7 (line 82 in `ai/agent.py`)
2. **Network initialization**: With random weight initialization, the sigmoid output of the cube decision head hovers around 0.5
3. **Cold start problem**: Since 0.5 < 0.7, no doubles are ever offered, so the network never learns to increase this value

**Evidence from your training:**
```
Doubles/game: 0.00, Accept rate: 0.00%
```

This persisted across all 10,000 matches, indicating the network never learned to use the doubling cube.

### Issue 2: Consistently High Win Rate for Player 1 (~90%)

**Problem:**
In self-play (same agent playing both sides), Player 1 wins ~90% of matches, rather than the expected ~50%.

**Possible Explanations:**

#### A) First-Move Advantage (Most Likely)
In backgammon, the player who moves first has a significant advantage:
- First chance to establish anchor points
- Can hit opponent's blots before they're secured
- Always one move ahead in a race situation

**Why this manifests in self-play:**
- Player 1 ALWAYS goes first in every game
- Even though both players use the same network, Player 1 consistently gets the positional advantage
- This could legitimately result in 60-70% win rate for first player

**Why you saw 90%:**
- Your trainer doesn't alternate which player goes first
- Line 110 in `training/match_trainer.py` creates new game with default starting player (always 1)
- Over a full match, the cumulative advantage compounds

#### B) Potential Network Evaluation Bias (Less Likely)
If the network evaluation isn't perfectly symmetric for both players, it could favor Player 1. However, the code shows proper player perspective handling via `board.encode_for_nn(player)`.

### Issue 3: No Learning Signal for Cube Decisions

**Problem:**
The cube decision head has two outputs (should_double, should_accept), but neither receives meaningful training signals.

**Why:**
- Training updates only occur based on match equity TD errors
- Since no doubles are offered, the cube decision outputs are never used
- The network has no incentive to change these outputs from their initial ~0.5 values

## The Fixes

I've created an improved trainer (`training/match_trainer_v2.py`) with the following enhancements:

### Fix 1: Dynamic Doubling Threshold

**Old approach:**
```python
threshold = 0.7  # Fixed threshold
```

**New approach:**
```python
# Start with lower threshold, gradually increase
threshold_start = 0.55
threshold_end = 0.70

# Linear interpolation over training
threshold = 0.55 + (match_num / total_matches) * 0.15
```

**Rationale:**
- Start at 0.55 so initial network outputs (~0.5) have a chance to trigger doubles
- Gradually increase to 0.70 as the network learns
- This breaks the cold-start problem

### Fix 2: Exploration Bonus for Cube Decisions

**New feature:**
```python
# Add exploration bonus to double probability
exploration_bonus = 0.15  # Start
# ... decreases to 0.00 over training

should_double_prob = network_output + exploration_bonus
if should_double_prob > threshold:
    offer_double()
```

**Rationale:**
- Encourages early doubling even with untrained network
- Provides training examples for cube decision head
- Decreases over time as network learns

### Fix 3: Alternating First Player

**Old approach:**
```python
game = BackgammonGame()  # Always starts with Player 1
```

**New approach:**
```python
# Randomize who goes first in match
first_player = random.choice([1, -1])

# Alternate within match
for game_num in games:
    starter = first_player if game_num % 2 == 0 else -first_player
    play_game(starting_player=starter)
```

**Rationale:**
- Eliminates systematic first-move bias
- Provides balanced training data
- Allows measurement of actual first-player advantage

### Fix 4: Enhanced Statistics Tracking

**New metrics:**
```python
match_wins_as_first = {1: 0, -1: 0}  # Track wins when going first
first_player_advantage = wins_as_first / total_matches
```

**Rationale:**
- Measure first-player advantage separately from Player 1 vs Player 2 win rate
- Expected: ~55-60% advantage for first player in backgammon
- Helps diagnose if high win rate is due to first-move advantage or network bias

## How to Use the Fixed Trainer

### Option 1: Fresh Training
```bash
python train_match_play_v2.py --matches 10000 --lr 0.0005 --epsilon 0.1
```

### Option 2: Continue from Your Existing Model
```bash
python train_match_play_v2.py --matches 10000 --lr 0.0005 --epsilon 0.1 \\
    --load models/match_model_final.pth
```

### Expected Results

You should see output like:
```
Matches: 500, P1 match win rate: 0.523, 1st-player adv: 0.572
  Games: 3768, Doubles/game: 0.12, Accept rate: 45.2%
  Current threshold: 0.565, exploration: 0.135

Matches: 1000, P1 match win rate: 0.501, 1st-player adv: 0.581
  Games: 7617, Doubles/game: 0.24, Accept rate: 52.1%
  Current threshold: 0.580, exploration: 0.120

...

Matches: 10000, P1 match win rate: 0.495, 1st-player adv: 0.592
  Games: 75779, Doubles/game: 0.85, Accept rate: 68.3%
  Current threshold: 0.700, exploration: 0.000
```

**Key improvements:**
1. **Doubling activity**: Should see 0.1-1.0 doubles/game (vs 0.00 before)
2. **Balanced P1 win rate**: ~50% (vs 90% before)
3. **Measurable first-player advantage**: ~55-60% (realistic for backgammon)
4. **Accept rate increasing**: As network learns proper cube decisions

## Understanding the Metrics

### P1 Match Win Rate
- **Old**: 90% - Systematic bias
- **New**: ~50% - Both players equally strong (self-play)

### 1st-Player Advantage
- **New metric**: Percentage of matches won by whoever went first
- **Expected**: 55-60% in backgammon
- **Indicates**: How much first-move advantage affects outcomes

### Doubles/Game
- **Old**: 0.00 - Network never learned to double
- **New**: 0.5-1.0 - Realistic doubling frequency
- **Note**: High-level human play shows ~0.3-0.8 doubles per game

### Accept Rate
- **Old**: N/A (no doubles offered)
- **New**: Should increase from ~40% to ~70% as network learns
- **Indicates**: Network learning to accept good cubes, reject bad cubes

## Why These Fixes Work

### Cold Start → Warm Start
By starting with lower threshold + exploration bonus, we ensure doubles happen early, which provides training data for the cube decision head.

### Biased → Balanced
Alternating first player ensures both Player 1 and Player 2 get equal opportunities to go first, eliminating systematic bias.

### Hidden Bias → Visible Metrics
Separate tracking of "Player 1 wins" vs "first-player wins" reveals whether bias is due to:
- Player identity (network bug)
- Move order (game dynamics)

## Recommended Training Procedure

### Important: Continuation Training Issue

⚠️ **When continuing from a checkpoint, the doubling threshold and exploration bonus RESET to their starting values**, even though the network weights are loaded correctly. This causes excessive doubling at the start of continuation training.

**Solution:** Use fixed hyperparameters for continuation training.

1. **Initial training** (5,000 matches):
   ```bash
   python train_match_play_v2.py --matches 5000 --epsilon 0.15 --lr 0.001
   ```
   - Higher epsilon (move exploration)
   - Higher learning rate
   - **Dynamic** threshold: 0.55 → 0.70
   - **Dynamic** exploration bonus: 0.15 → 0.00
   - Focus on discovering doubling strategies

2. **Refinement training** (10,000 matches):
   ```bash
   python train_match_play_v2.py --matches 10000 --epsilon 0.1 --lr 0.0005 \
       --load models/match_model_5000.pth \
       --double-threshold 0.70 --exploration-bonus 0.00
   ```
   - ⭐ **CRITICAL:** Use `--double-threshold 0.70 --exploration-bonus 0.00` to avoid reset
   - Lower epsilon for moves
   - Lower learning rate
   - **Fixed** doubling parameters
   - Refine cube decisions

3. **Fine-tuning** (10,000 matches):
   ```bash
   python train_match_play_v2.py --matches 10000 --epsilon 0.05 --lr 0.0001 \
       --load models/match_model_15000.pth \
       --double-threshold 0.70 --exploration-bonus 0.00
   ```
   - Minimal epsilon
   - Very low learning rate
   - **Fixed** doubling parameters
   - Polish final strategy

## Expected Training Time

Based on your previous run (10,000 matches in ~15.7 hours):
- **Initial**: 5,000 matches → ~8 hours
- **Refinement**: 10,000 matches → ~16 hours
- **Fine-tuning**: 10,000 matches → ~16 hours
- **Total**: ~40 hours for complete training

## Questions?

If you see:
- **Still no doubling**: Check that you're using `train_match_play_v2.py`, not the old script
- **Still 90% P1 win rate**: Check the "1st-player adv" metric - if that's also 90%, it's first-move advantage
- **Network diverging**: Lower learning rate
- **Too much doubling**: Increase threshold_start in the trainer
