# Tetris AI Agent

A collection of reinforcement learning and search-based agents for playing Tetris using the Tetris-Gymnasium environment.

## Overview

This project implements multiple AI strategies for playing Tetris:

- **ES Look-Ahead**: Evolution Strategy with lookahead capability examining 1-4 tetrominoes in the queue
- **MCTS**: Monte Carlo Tree Search implementation  
- **Beam Search**: Beam search with configurable width and lookahead depth
- **Value Function Learning**: Deep Q-Network style value function training
- **Conservative SAC**: Soft Actor-Critic with conservative updates using transformer backbone

## Requirements

- Python 3.8+
- gymnasium
- numpy
- matplotlib
- torch (for neural network agents)
- wandb (for experiment tracking)
- stable-baselines3 (for DQN agent)
- hydra-core (for configuration management)

## Installation

1. Install the Tetris-Gymnasium environment:
```bash
cd Tetris-Gymnasium
pip install -e .
```

2. Install additional dependencies:
```bash
pip install torch wandb stable-baselines3 hydra-core matplotlib
```

## Usage

### ES Look-Ahead Agent

Run the ES strategy with different lookahead depths and search strategies:

```bash
# Basic lookahead with 2 tetrominoes
python ES_Look_Ahead.py --num_look_ahead 2 --strategy lookahead

# MCTS with time limit
python ES_Look_Ahead.py --strategy mcts --mcts_time_limit 5.0 --num_look_ahead 3

# Beam search with custom width
python ES_Look_Ahead.py --strategy beam --beam_width 15 --num_look_ahead 2
```

**Parameters:**
- `--num_look_ahead`: Number of tetrominoes to look ahead (1-4)
- `--strategy`: Strategy to use (`lookahead`, `mcts`, `beam`, `heuristic`)
- `--beam_width`: Beam width for beam search (1-40)
- `--mcts_rollouts`: Number of MCTS rollouts per action
- `--mcts_time_limit`: Time limit for MCTS in seconds
- `--mcts_exploration_weight`: UCB exploration weight for MCTS

### Value Function Training

Train a value function using temporal difference learning:

```bash
python run_value.py --num_epochs 3000 --lr 0.001 --gamma 0.99
```

**Parameters:**
- `--gamma`: Discount factor
- `--lr`: Learning rate
- `--init_epsilon`: Initial exploration rate
- `--final_epsilon`: Final exploration rate
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs
- `--eval_freq`: Evaluation frequency
- `--tau`: Soft update rate for target network

### DQN Training with Hydra

Train a DQN agent using Stable Baselines3 and Hydra configuration:

```bash
python train_agent.py
```

Configuration can be modified in `conf/dqn.yaml`.

### Conservative SAC Training

Train a Conservative SAC agent with transformer backbone:

```bash
python conservative_sac_main.py --env=tetris-v0 --n_epochs=1000
```

## File Structure

```
Tetris_Agent/
├── ES_Look_Ahead.py          # Main lookahead/MCTS/beam search agent
├── run_value.py              # Value function training with DQN-style learning
├── train_agent.py            # DQN training with Stable Baselines3
├── conservative_sac_main.py  # Conservative SAC with transformer backbone
├── conf/                     # Hydra configuration files
│   ├── dqn.yaml
│   └── logging/
│       └── common.yaml
└── README.md
```

## Features

### Search Strategies

1. **Multistep Lookahead**: Recursively evaluates all possible placements for upcoming tetrominoes
2. **MCTS**: Monte Carlo Tree Search with UCB node selection and random rollouts
3. **Beam Search**: Maintains top-k nodes at each search depth level
4. **Feature-based Evaluation**: Uses height, holes, bumpiness, and completed lines as heuristics

### Neural Network Agents

1. **Value Function**: Learns state values using feature vectors and TD learning
2. **DQN**: Deep Q-Network using Stable Baselines3 implementation  
3. **Conservative SAC**: Soft Actor-Critic with conservative Q-learning and transformer backbone

### Evaluation Metrics

All agents log performance metrics including:
- Total reward per episode
- Episode length
- Evaluation scores
- Training loss (for neural network agents)

## Experiment Tracking

The project uses Weights & Biases (wandb) for experiment tracking. Make sure to configure your wandb account before running training scripts.

## Contributing

This is a personal project for experimenting with different AI approaches to Tetris. Feel free to use and modify the code for your own experiments.

## License

This project uses the Tetris-Gymnasium environment. Please refer to the original repository for licensing information.