# Examples

This directory contains examples demonstrating the use of swtoolkit's ML infrastructure.

## Tic Tac Toe Example

The Tic Tac Toe example demonstrates how to use the `Evaluator` and `DataSaver` classes for reinforcement learning.

### Quick Start

Play 100 games with random agents and collect training data:

```bash
python examples/tic_tac_toe.py --games 100
```

### Command Line Options

- `--model1 PATH`: Path to model file for player 1 (default: random)
- `--model2 PATH`: Path to model file for player 2 (default: random)
- `--games N`: Number of games to play (default: 10)
- `--save-dir DIR`: Directory to save training data (default: tic_tac_toe_data)
- `--save-frequency N`: Save data every N experiences (default: 50)
- `--gamma FLOAT`: Discount factor for reward propagation (default: 0.9)
- `--display`: Display each game as it's played
- `--no-data`: Disable data collection

### Examples

Play 1000 games with data collection:
```bash
python examples/tic_tac_toe.py --games 1000 --save-frequency 100
```

Watch a single game with display:
```bash
python examples/tic_tac_toe.py --games 1 --display
```

Play without collecting data:
```bash
python examples/tic_tac_toe.py --games 100 --no-data
```

### Understanding the Infrastructure

#### Evaluator Class

The `Evaluator` class handles loading models and getting actions:

```python
from swtoolkit.ml import Evaluator

# Create evaluator (None = random actions)
evaluator = Evaluator(model_path=None)

# Get action for current state
state = [0, 1, 0, 2, 0, 0, 0, 0, 0]  # Tic-tac-toe board
valid_actions = [0, 2, 4, 5, 6, 7, 8]  # Empty positions
action = evaluator.get_action(state, valid_actions)
```

#### DataSaver Class

The `DataSaver` class manages training data collection:

```python
from swtoolkit.ml import DataSaver

# Create data saver
saver = DataSaver(
    save_dir="training_data",
    save_frequency=100,  # Save every 100 experiences
    gamma=0.9  # Discount factor for reward propagation
)

# Add experience
exp_id = saver.add_experience(
    state=current_state,
    action=action_taken,
    reward=0.0,
    next_state=next_state,
    done=False
)

# Update reward later (e.g., at end of episode)
saver.set_reward(exp_id, reward=1.0, propagate=True)

# End episode
saver.end_episode()

# Force save
saver.flush()
```

### Data Format

Training data is saved as pickle files in the specified directory:
- `experiences_XXXXXXXX_YYYYYYYY.pkl`: Binary pickle format
- `experiences_XXXXXXXX_YYYYYYYY.json`: Human-readable JSON format

Each experience contains:
- `id`: Unique experience ID
- `state`: State before action
- `action`: Action taken
- `reward`: Reward received (with gamma propagation if enabled)
- `next_state`: State after action
- `done`: Whether episode ended
- Additional custom fields

### Reward Propagation

When `gamma` is set (e.g., 0.9), rewards are propagated backwards to previous actions in the episode:

```
If final reward = 1.0 and gamma = 0.9:
  Last action:     reward = 1.0
  2nd last action: reward = 0.9
  3rd last action: reward = 0.81
  4th last action: reward = 0.729
  ...
```

This helps credit earlier actions that led to the final outcome.
