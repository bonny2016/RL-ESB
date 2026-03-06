# Electric Bus Scheduling Optimization with Deep Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Deep Reinforcement Learning solution for optimizing electric bus fleet scheduling with multiple state-of-the-art on-policy & off-policy algorithms. This project implements and compares four different RL approaches: **PPO**, **DDQN**, **REINFORCE**, and **A2C** for solving the electric bus scheduling problem.

## Project Lineage

This repository is inherited from the original project:
- Source: [https://github.com/meghkc/RL-ESB](https://github.com/meghkc/RL-ESB)

The original codebase provides the baseline RL environment, agents, and training scripts for a synthetic electric bus scheduling setup.

## What Changed In This Repo

This fork extends the original project with dataset-driven scheduling support and related training updates:

- Added `dataArticleJuliette/` support (Montreal transit benchmark instance format).
- Added `dataset_loader.py` to parse `depots.txt`, `recharge.txt`, `voyages.txt`, and `hlp.txt`.
- Updated `environment.py` to support both:
  - `synthetic` mode (original config-based setup), and
  - `juliette` mode (instance files from `dataArticleJuliette`).
- Updated `run.py` with CLI options for dataset selection:
  - `--data-source`, `--dataset-root`, `--dataset-subset`, `--dataset-split`, `--dataset-instance`.
- Updated PPO flow (`run.py`, `ppo_agent.py`) to use environment-provided valid actions and dynamic state/action dimensions.
- Updated setup notes so only core PyTorch is required by default (no mandatory `torchvision`/`torchaudio`).

## ðŸŽ¯ Features

- **Multi-Algorithm Implementation**: Compare 4 different RL approaches
  - **PPO (Proximal Policy Optimization)**: Stable policy gradient method
  - **DDQN (Double Deep Q-Network)**: Advanced value-based method
  - **REINFORCE**: Classic policy gradient algorithm
  - **A2C (Advantage Actor-Critic)**: Actor-critic with advantage estimation
- **GPU/CPU Support**: Automatic hardware detection and optimization
- **Comprehensive Comparison**: Side-by-side algorithm performance analysis
- **Real-world Constraints**: Multi-line bus scheduling with time constraints
- **Action Masking**: Intelligent constraint handling for valid bus assignments
- **Reward Engineering**: Multi-objective optimization (fleet size, deadhead costs, chain bonuses)

## ðŸš€ Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib pandas tqdm
# Optional only if your project code needs them:
# pip install torchvision torchaudio
```

### Installation

```bash
git clone https://github.com/meghkc/RL-ESB.git
cd RL-ESB
pip install -r requirements.txt
```

### Usage

#### 1. Train Individual Algorithms

```bash
# Train PPO
python run.py

# Train DDQN  
python run_ddqn.py

# Train REINFORCE
python run_reinforce.py

# Train A2C
python run_a2c.py
```

#### 2. Compare All Algorithms

```bash
# Run comparison analysis and generate plots
python compare_algorithms.py
```

#### 3. Algorithm-Specific Training Scripts

```bash
# PPO with custom parameters
python run.py --episodes 1000

# DDQN with exploration tuning
python run_ddqn.py

# REINFORCE policy gradient
python run_reinforce.py  

# A2C actor-critic
python run_a2c.py
```

## ðŸ“Š Project Structure

```
RL-ESB/
â”œâ”€â”€ config.py                    # System configuration and hyperparameters
â”œâ”€â”€ environment.py               # Bus scheduling environment implementation
â”œâ”€â”€ 
â”œâ”€â”€ # Algorithm Implementations
â”œâ”€â”€ ppo_agent.py                 # Proximal Policy Optimization agent
â”œâ”€â”€ ddqn_agent.py               # Double Deep Q-Network agent  
â”œâ”€â”€ reinforce_agent.py          # REINFORCE policy gradient agent
â”œâ”€â”€ a2c_agent.py                # Advantage Actor-Critic agent
â”œâ”€â”€
â”œâ”€â”€ # Training Scripts  
â”œâ”€â”€ run.py                      # PPO training and evaluation
â”œâ”€â”€ run_ddqn.py                 # DDQN training and evaluation
â”œâ”€â”€ run_reinforce.py            # REINFORCE training and evaluation
â”œâ”€â”€ run_a2c.py                  # A2C training and evaluation
â”œâ”€â”€ compare_algorithms.py       # Multi-algorithm comparison and analysis
â”œâ”€â”€
â”œâ”€â”€ data/                       # Results and model storage
â”‚   â”œâ”€â”€ *_training.txt          # Training progress logs
â”‚   â”œâ”€â”€ *_results.txt           # Evaluation results
â”‚   â”œâ”€â”€ *_best_model.pth        # Best trained models
â”‚   â””â”€â”€ figures/                # Training plots and comparisons
â”‚       â”œâ”€â”€ *_training_rewards.png
â”‚       â””â”€â”€ algorithm_comparison.png
â”œâ”€â”€
â”œâ”€â”€ requirements.txt            # Python dependencies
â””â”€â”€ README.md                   # This file
```

## ðŸ› ï¸ Configuration

Edit `config.py` to customize:

- **Bus Lines**: Multiple bus routes with different terminals and schedules
- **Operation Parameters**: Operating hours (6 AM - 9 PM), trip times, rest periods
- **Fleet Settings**: Number of available buses, depot configuration
- **Reward Weights**: Deadhead costs, unused bus penalties, chain bonuses
- **Algorithm Hyperparameters**: Learning rates, network architectures, training episodes

## ðŸ† Algorithm Comparison

| Algorithm | Type | Key Features | Best For |
|-----------|------|--------------|----------|
| **PPO** | Policy Gradient | Stable, clipped updates | General purpose, stable training |
| **DDQN** | Value-based | Experience replay, target network | Sample efficiency, discrete actions |
| **REINFORCE** | Policy Gradient | Simple, direct optimization | Understanding baselines |
| **A2C** | Actor-Critic | Advantage estimation, shared features | Reduced variance, faster convergence |

## ðŸ“ˆ Results

The system learns to:
- **Minimize Fleet Size**: Use fewer buses while meeting all trip demands
- **Optimize Scheduling**: Reduce deadhead movements between terminals  
- **Create Efficient Chains**: Assign consecutive trips to same buses on same lines
- **Handle Constraints**: Respect bus availability and time windows
- **Balance Trade-offs**: Fleet size vs. operational efficiency

### Performance Metrics
- **Average Reward**: Overall system performance
- **Buses Used**: Fleet efficiency (typically 3-5 buses from 10 available)
- **Training Stability**: Convergence speed and consistency
- **Solution Quality**: Meeting all trip demands with minimal resources

## ðŸ”¬ Technical Details

### Problem Definition
- **3 Bus Lines**: Different terminals (Terminal1, Terminal2, Terminal3) with varying schedules
- **Operation Period**: 15-hour daily operation (6:00 AM - 9:00 PM)
- **Fleet Management**: 10 available buses, minimize actual usage
- **Scheduling Constraints**: Bus availability, terminal locations, trip timing

### Environment State Space
- **Time Information**: Normalized current event time and bus line ID
- **Bus Status**: Availability times for each bus (positive = available, negative = busy)
- **Location Tracking**: Current terminal positions for deadhead cost calculation

### Action Space
- **Bus Assignment**: Select which bus (0-9) to assign to each trip
- **Action Masking**: Only available buses can be selected
- **Constraint Handling**: Automatic filtering of invalid assignments

### Reward Function Components
- **Deadhead Cost**: Penalty for moving empty buses between terminals
- **Unused Bus Penalty**: Discourage using new buses when used buses are available  
- **Rest Reward**: Bonus for reusing available buses
- **Chain Bonus**: Extra reward for consecutive trips on same line
- **Final Penalty**: Total fleet size minimization at episode end

### Algorithm Implementations

#### PPO (Proximal Policy Optimization)
- **Architecture**: Shared state network + separate actor/critic heads
- **Features**: Action masking, GAE advantages, clipped policy updates
- **Stability**: Trust region optimization for consistent learning

#### DDQN (Double Deep Q-Network)  
- **Architecture**: Separate main and target Q-networks
- **Features**: Experience replay, Îµ-greedy exploration, target network updates
- **Efficiency**: Sample reuse and stable value estimation

#### REINFORCE
- **Architecture**: Simple policy network with softmax output
- **Features**: Monte Carlo returns, advantage normalization
- **Simplicity**: Direct policy gradient optimization

#### A2C (Advantage Actor-Critic)
- **Architecture**: Shared base + actor/critic heads
- **Features**: TD advantage estimation, simultaneous policy/value updates
- **Balance**: Reduced variance while maintaining policy gradient benefits

## ï¿½ Hardware Support

All algorithms automatically detect and utilize available hardware:

- **GPU Acceleration**: CUDA support for faster training
- **CPU Fallback**: Automatic CPU usage when GPU unavailable  
- **Device Logging**: Clear indication of hardware being used
- **Memory Optimization**: Efficient tensor operations and batch processing

## ðŸ“Š Visualization and Analysis

The project generates comprehensive visualizations:

- **Individual Training Curves**: Progress for each algorithm
- **Comparative Analysis**: Side-by-side performance comparison
- **Performance Metrics**: Detailed evaluation statistics
- **Solution Quality**: Bus usage efficiency and constraint satisfaction

## ï¿½ðŸ“š Documentation

For detailed documentation, see:
- Code comments and docstrings in each module
- Algorithm-specific implementation details
- Environment dynamics and reward engineering
- Comparison methodology and metrics

## ðŸ¤ Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ðŸ“„ License

This project is licensed under the MIT License.

## ðŸ“ž Contact

Project Link: [https://github.com/meghkc/RL-ESB](https://github.com/meghkc/RL-ESB)

## ðŸ™ Acknowledgments

- Built with PyTorch for deep learning and GPU acceleration
- Inspired by advances in deep reinforcement learning for transportation optimization
- Algorithm implementations based on seminal papers in RL
- Thanks to the open-source RL community for foundational work
- Multi-algorithm comparison methodology for comprehensive evaluation

