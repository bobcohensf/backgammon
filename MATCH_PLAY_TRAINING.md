# Match Play Training with Cube Decisions

This document explains the match-aware training system that teaches the AI to:
- **Optimize for match winning** (not just individual games)
- **Make cube decisions** (when to double, when to accept/reject)
- **Consider match score context** (being ahead/behind changes strategy)

## Overview

### Key Improvements Over Single-Game Training

1. **Match Equity Optimization**: The AI learns to maximize the probability of winning the **match**, not just individual games
2. **Cube Decision Training**: Learns when to offer doubles and when to accept/reject doubles
3. **Context-Aware Strategy**: Considers current match score, making different decisions when ahead vs behind
4. **Crawford Rule**: Understands and respects the Crawford game (no doubling when leader is 1 point away)

## Architecture

### MatchAwareBackgammonNet (`ai/network.py`)

Enhanced neural network with two output heads:

**Inputs:**
- Board features (196 dimensions) - same as before
- Match context (5 dimensions):
  - My score / match length
  - Opponent score / match length
  - Cube value / 64
  - Crawford indicator (0 or 1)
  - Cube owner (-1, 0, or 1)

**Outputs:**
1. **Match Equity Head**: Probability of winning the match (0-1)
2. **Cube Decision Head**: Two values:
   - `should_double`: Confidence score for offering a double
   - `should_accept`: Confidence score for accepting a double

### MatchAwareTDAgent (`ai/agent.py`)

Enhanced agent with match-aware decision making:

```python
# Move selection considers match score
agent.select_move(game, player, dice, my_score, opp_score, is_crawford)

# Cube decisions based on learned strategy
agent.should_offer_double(board, player, my_score, opp_score, is_crawford)
agent.should_accept_double(board, player, my_score, opp_score, is_crawford)
```

### MatchPlayTrainer (`training/match_trainer.py`)

Trains through complete matches:
- Plays matches to N points (default 7)
- Includes doubling cube actions during games
- Uses match equity as reward signal
- Enforces Crawford rule
- Tracks detailed statistics

## Training

### Quick Start

```bash
# Train a new match-aware model
python train_match_play.py --matches 5000 --match-length 7

# Continue training from checkpoint
python train_match_play.py --matches 5000 --load models/match_model_final.pth

# Train with custom hyperparameters
python train_match_play.py \
    --matches 10000 \
    --lr 0.0003 \
    --epsilon 0.2 \
    --lambda 0.7 \
    --hidden-sizes 512 256 128
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--matches` | 5000 | Number of matches to play |
| `--match-length` | 7 | Points to win each match |
| `--lr` | 0.0005 | Learning rate (lower than single-game training) |
| `--epsilon` | 0.15 | Exploration rate |
| `--lambda` | 0.7 | TD(λ) parameter |
| `--save-every` | 500 | Save checkpoint frequency |
| `--hidden-sizes` | [256, 128, 64] | Network architecture |

### What the AI Learns

During training, the AI learns:

1. **Position Evaluation in Match Context**
   - Being up 6-4 in a match to 7 changes the value of aggressive plays
   - Trailing players need to take more risks
   - Leading players can play more conservatively

2. **Cube Timing**
   - When position advantage is strong enough to double
   - When to accept a cube even in difficult positions (match equity considerations)
   - When to reject and concede the game to preserve match score

3. **Match Situations**
   - DMP (Double Match Point): Both players at match point
   - Leading strategies: Protect the lead
   - Trailing strategies: Take calculated risks

## Using the Trained Model

### In the Web Interface

The web server automatically loads the match-aware model if available:

```python
# Priority order:
1. models/match_model_final.pth  (match-aware model)
2. models/model_final.pth        (single-game model)
3. Untrained model
```

**To use your trained model:**
```bash
# After training completes, the model is automatically saved
# Just run the web server:
python web_server.py
```

The AI will:
- Make moves optimized for match equity
- Offer doubles when appropriate
- Accept/reject doubles based on match score and position

### Manual Model Selection

If you have multiple models:

```bash
# Rename to use specific model
cp models/match_model_5000.pth models/match_model_final.pth

# Or for single-game model
cp models/model_final.pth models/model_backup.pth
```

## Training Tips

### Recommended Training Progression

1. **Stage 1: Basic Position Learning** (2000-3000 matches)
   - Higher epsilon (0.2) for exploration
   - Learn basic position evaluation
   - Cube decisions will be random initially

2. **Stage 2: Cube Decision Refinement** (3000-5000 matches)
   - Lower epsilon (0.1) to exploit learned patterns
   - Cube decisions become more strategic

3. **Stage 3: Fine-tuning** (5000+ matches)
   - Very low epsilon (0.05)
   - Polish match equity evaluation
   - Refine cube decisions for edge cases

### Example Progressive Training

```bash
# Stage 1: Exploration
python train_match_play.py --matches 3000 --epsilon 0.2 --lr 0.001

# Stage 2: Exploitation (continue from stage 1)
python train_match_play.py --matches 3000 --epsilon 0.1 --lr 0.0005 \
    --load models/match_model_final.pth

# Stage 3: Fine-tuning
python train_match_play.py --matches 4000 --epsilon 0.05 --lr 0.0003 \
    --load models/match_model_final.pth
```

## Comparison with Single-Game Training

| Aspect | Single-Game Training | Match Play Training |
|--------|---------------------|---------------------|
| **Objective** | Win individual games | Win matches |
| **Reward Signal** | Game outcome (0 or 1) | Match equity (0-1) |
| **Cube Decisions** | Not trained | Fully integrated |
| **Match Context** | Ignored | Central to strategy |
| **Training Time** | Faster (simpler) | Slower (more complex) |
| **Play Strength** | Good at individual games | Better at match strategy |

## Monitoring Training

Watch these metrics during training:

```
Matches: 2000, P1 match win rate: 0.485
  Games: 15432, Doubles/game: 0.23, Accept rate: 67.3%
```

**Good signs:**
- Match win rate converging to ~0.50 (balanced)
- Doubles/game increasing (learning to use cube)
- Accept rate 50-75% (discriminating well)

**Red flags:**
- Accept rate >95% (accepting too much)
- Accept rate <25% (rejecting too much)
- Doubles/game <0.05 (not using cube)

## Technical Details

### TD Learning for Match Equity

The trainer uses TD(λ) to learn match equity:

```python
# For each position during a match:
current_match_equity = network.predict(board, match_context)
next_match_equity = network.predict(next_board, next_match_context)

# TD error
td_error = next_match_equity - current_match_equity

# Terminal state
final_match_equity = 1.0 if won_match else 0.0
```

### Cube Decision Training

Cube decisions are trained implicitly:
- When AI offers double and opponent rejects → learns position was strong
- When AI accepts and loses badly → learns to be more selective
- Match score context teaches when cube leverage is high/low

## Performance Expectations

After 5000-10000 matches of training:

- **Move Quality**: Similar to single-game training
- **Cube Usage**: 0.15-0.30 doubles per game
- **Cube Accuracy**: 60-75% of doubles are "correct" by expert standards
- **Match Win Rate**: 45-55% against itself (balanced)
- **vs Untrained**: 70-80% match win rate

## Future Enhancements

Potential improvements:

1. **Match Equity Tables**: Pre-computed tables for endgame situations
2. **Opening Book**: Learn match-specific opening strategies
3. **Cube Action Separation**: Separate networks for doubling and take/pass
4. **Position Evaluation**: Additional heads for gammon/backgammon probabilities
5. **Opponent Modeling**: Adapt to opponent's cube handling style

## Troubleshooting

**Problem: AI never doubles**
- Solution: Increase exploration (higher epsilon), train longer

**Problem: AI accepts all doubles**
- Solution: Reduce accept threshold, ensure diverse training positions

**Problem: Training very slow**
- Solution: Reduce match length during training, use smaller network

**Problem: Cube decisions seem random**
- Solution: More training needed (cube decisions are harder than moves)

## References

- Match equity theory from backgammon literature
- TD-Gammon architecture (Tesauro, 1995)
- Modern backgammon cube handling principles

---

**Note**: This is an advanced feature requiring significant computational resources. For casual play, the single-game model may be sufficient. For serious match play, invest in extended training.
