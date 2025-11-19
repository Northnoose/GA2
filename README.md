# PyTorch Deep Q-Learning for Snake (v17.1)
This project re-implements the original TensorFlow Snake DQN agent in PyTorch, following the exact API and behaviour from the DTE-2502 assignment.
It includes a full training pipeline, replay buffer, logging, and GIF-based visualization.e.

## The system includes:

- PyTorch DQN model
- Modular training loop0
- Replay buffer
- GIF visualization of trained policies
- Same API as the original TensorFlow version

project/
│
├── agent.py                # DeepQLearningAgent (PyTorch)
├── training.py             # Full training pipeline
├── replay_buffer.py        # Replay buffer implementation
├── utils.py                # Support functions
├── visualize_policy_gif.py # Policy visualization
│
├── models/                 # Saved .pt model files
├── images/                 # Generated GIFs
└── logs/                   # CSV training logs

## DQN Model Architecture
The DQN model is implemented using PyTorch and matches the 17.1 assignment architecture:

- 3 convolutional layers (16, 32, 64 channels)
- ReLU activation
- Flatten layer
- Fully connected (64 → n_actions)
This matches version 17.1 of the assignment architecture.

## Agent
The DeepQLearningAgent uses:

- Huber loss
- RMSprop optimizer
- Hard target network sync
- Epsilon-greedy action selection
- Optional reward clipping

## Training
Run Training: python training.py
Outputs:
- models/ – saved .pt model files
- logs/ – CSV training logs

CSV contains:
- reward mean
- length mean
- games played
- loss

## Visualization
Generate GIF: python visualize_policy_gif.py
Output is saved in images/

## Install
pip install -r requirements.txt

Requires Python 3.9+ and PyTorch.

## Results
- Reward improves gradually but remains unstable
- Snake length follows the same pattern
- Loss decreases but spikes 

## Limitations
- Only 50k iterations (needs ~200k for stable DQN)
- No Double-DQN
- No prioritized replay
- Sparse rewards

## Future Improvements
- Add Double-DQN
- Add Prioritized Replay
- Train longer
- Improve reward shaping

## PowerPoint
If your looking for the presentation, here is a link to accses the PPT
https://drive.google.com/drive/folders/1TQ7JsVr5fEvNWYBDOSAqt5kcCAmfBNSe?usp=drive_link
