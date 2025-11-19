'''
PyTorch-treningsskript for Snake DQN (v17.1-konfig).
Beholder eksisterende CLI-oppførsel/struktur i grove trekk.
'''

import os
import time
import json
import numpy as np
import pandas as pd

from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
from agent import DeepQLearningAgent, PolicyGradientAgent, AdvantageActorCriticAgent

# ----------------------------
# Konfigurasjon
# ----------------------------
version = 'v17.1'
with open(f'model_config/{version}.json', 'r') as f:
    m = json.loads(f.read())
board_size = m['board_size']
frames = m['frames']  # >= 2
max_time_limit = m['max_time_limit']
supervised = bool(m['supervised'])
n_actions = m['n_actions']
obstacles = bool(m['obstacles'])
buffer_size = m['buffer_size']
games = 512  # for numpy-miljøet

# Treningsparametre
episodes = 50000
log_frequency = 1000
games_eval = 8

# ----------------------------
# Agent
# ----------------------------
agent = DeepQLearningAgent(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=buffer_size,
    version=version,
)
# agent = PolicyGradientAgent(...)
# agent = AdvantageActorCriticAgent(...)

# Agent-type streng for eksisterende kontrollflyt
if isinstance(agent, DeepQLearningAgent):
    agent_type = 'DeepQLearningAgent'
elif isinstance(agent, PolicyGradientAgent):
    agent_type = 'PolicyGradientAgent'
elif isinstance(agent, AdvantageActorCriticAgent):
    agent_type = 'AdvantageActorCriticAgent'
else:
    agent_type = 'Unknown'

print('Agent is {:s}'.format(agent_type))

# ----------------------------
# Epsilon/utforskning for DQN
# ----------------------------
if agent_type in ['DeepQLearningAgent']:
    epsilon, epsilon_end = 1.0, 0.03
    reward_type = 'current'
    sample_actions = False
    n_games_training = games
    decay = 0.997
    if supervised:
        epsilon = 0.01
        try:
            agent.load_model(file_path=f'models/{version}', iteration=1)
        except FileNotFoundError:
            pass

if agent_type in ['PolicyGradientAgent']:
    epsilon, epsilon_end = -1, -1
    reward_type = 'discounted_future'
    sample_actions = True
    n_games_training = 16
    decay = 1

if agent_type in ['AdvantageActorCriticAgent']:
    epsilon, epsilon_end = -1, -1
    reward_type = 'current'
    sample_actions = True
    n_games_training = 32
    decay = 1

# ----------------------------
# Init miljøer (numpy-parallell for trening + vanlig for evaluering)
# ----------------------------
if agent_type in ['DeepQLearningAgent']:
    if supervised:
        try:
            agent.load_model(file_path=f'models/{version}', iteration=1)
        except FileNotFoundError:
            pass
    else:
        games = 512
        env = SnakeNumpy(
            board_size=board_size,
            frames=frames,
            max_time_limit=max_time_limit,
            games=games,
            frame_mode=True,
            obstacles=obstacles,
            version=version,
        )
        env2 = SnakeNumpy(
            board_size=board_size,
            frames=frames,
            max_time_limit=max_time_limit,
            games=games_eval,
            frame_mode=True,
            obstacles=obstacles,
            version=version,
        )

# ----------------------------
# Logging
# ----------------------------
model_logs = {'iteration': [], 'reward_mean': [], 'length_mean': [], 'games': [], 'loss': []}
os.makedirs('model_logs', exist_ok=True)
os.makedirs(f'models/{version}', exist_ok=True)

# ----------------------------
# Treningsløkke
# ----------------------------
loss = 0.0
total_games = None

for index in range(episodes):
    if agent_type in ['DeepQLearningAgent']:
        # Samle erfaring og tren
        _, _, _ = play_game2(
            env,
            agent,
            n_actions,
            epsilon=epsilon,
            n_games=n_games_training,
            record=True,
            sample_actions=sample_actions,
            reward_type=reward_type,
            frame_mode=True,
            total_frames=1500,
            stateful=True,
        )
        # Ett vanlig treningssteg
        loss = agent.train_agent(batch_size=32, num_games=n_games_training, reward_clip=False)

# Tre ekstra treningssteg for å utnytte CPU
        for _ in range(3):
            agent.train_agent(batch_size=32, reward_clip=False)


    # Eval/logg
    if (index + 1) % log_frequency == 0:
        current_rewards, current_lengths, current_games = play_game2(
            env2,
            agent,
            n_actions,
            n_games=games_eval,
            epsilon=-1,
            record=False,
            sample_actions=False,
            frame_mode=True,
            total_frames=-1,
            total_games=games_eval,
        )
        model_logs['iteration'].append(index + 1)
        model_logs['reward_mean'].append(round(int(current_rewards) / current_games, 2))
        model_logs['length_mean'].append(round(int(current_lengths) / current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']] \
            .to_csv(f'model_logs/{version}.csv', index=False)

        # Oppdater target + lagre
        agent.update_target_net()
        agent.save_model(file_path=f'models/{version}', iteration=(index + 1))
        epsilon = max(epsilon * decay, epsilon_end)
