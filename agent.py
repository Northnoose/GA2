"""
Minimal agent implementations for the project.

This module provides a full PyTorch implementation of DeepQLearningAgent
with the same public API as the original TF/Keras version
(move, add_to_buffer, train_agent, update_target_net, save_model,
load_model, get_gamma, get_buffer_size).

The remaining agent classes are stubs so that imports from the original
codebase still work, even though they are not implemented here.
"""

import os
import json
import time
import numpy as np
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer, ReplayBufferNumpy  # existing buffer implementations


# ----------------------------
# Base and stub agent classes
# ----------------------------
class Agent:
    """Base agent type used only for type consistency in this project."""
    pass


class PolicyGradientAgent(Agent):
    """Stub for original Policy Gradient agent (not implemented in this PyTorch version)."""
    pass


class AdvantageActorCriticAgent(Agent):
    """Stub for original Advantage Actor-Critic agent (not implemented here)."""
    pass


class HamiltonianCycleAgent(Agent):
    """Stub for Hamiltonian cycle based agent (not implemented in this project version)."""
    pass


class BreadthFirstSearchAgent(Agent):
    """Stub for BFS-based agent (not implemented in this project version)."""
    pass


# ----------------------------
# PyTorch DQN network
# ----------------------------
class DQN(nn.Module):
    """
    Convolutional DQN network mapping board states to Q-values.

    The network expects input in NCHW format: (batch, channels=frames,
    height=board_size, width=board_size). The architecture mirrors the
    configuration defined in v17.1.json (three conv blocks followed by
    a fully connected head with 64 hidden units).
    """
    def __init__(self, in_channels: int, n_actions: int, board_size: int):
        super().__init__()
        # Convolutional feature extractor: 3x3 → 5x5 → 5x5 layers with
        # padding chosen to preserve spatial dimensions.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        # Flatten and project to Q-values for each action.
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(64 * board_size * board_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input batch with shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Q-values for each action with shape (N, n_actions).
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


# ----------------------------
# DeepQLearningAgent (PyTorch)
# ----------------------------
class DeepQLearningAgent(Agent):
    """
    PyTorch-based Deep Q-Learning agent that mirrors the API and behaviour
    of the original TensorFlow implementation.

    Public interface used by the rest of the project:
    - move(board, legal_moves, value=None)
    - add_to_buffer(...)
    - train_agent(batch_size=..., ...)
    - update_target_net()
    - save_model(file_path, iteration)
    - load_model(file_path, iteration)
    - get_gamma()
    - get_buffer_size()
    """
    def __init__(
        self,
        board_size: int = 10,
        frames: int = 2,
        buffer_size: int = 10000,
        gamma: float = 0.99,
        n_actions: int = 4,
        use_target_net: bool = True,
        version: str = "v17.1",
        lr: float = 5e-4,
        device: Optional[str] = None,
    ):
        super().__init__()
        self._board_size = board_size
        self._frames = frames
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = bool(use_target_net)
        self._version = version
        self._reward_clip = False

        # Device selection (CPU by default, GPU if explicitly requested).
        if device is None:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Experience replay buffer using the existing numpy-based implementation.
        self._buffer = ReplayBufferNumpy(
            buffer_size=buffer_size,
            board_size=board_size,
            frames=frames,
            actions=n_actions,
        )

        # Online Q-network.
        self._model = DQN(
            in_channels=frames,
            n_actions=n_actions,
            board_size=board_size
        ).to(self._device)

        # Optional target network for more stable Q-learning.
        if self._use_target_net:
            self._target_net = DQN(
                in_channels=frames,
                n_actions=n_actions,
                board_size=board_size
            ).to(self._device)
            self.update_target_net()  # initial hard sync
        else:
            self._target_net = None

        # Optimizer and loss function (Huber loss).
        self._optimizer = optim.RMSprop(self._model.parameters(), lr=lr)
        self._criterion = nn.SmoothL1Loss()

        # Cache for the most recent training loss.
        self._last_loss = 0.0

    # ----------------------------
    # Public API used by utils/training/visualisation
    # ----------------------------
    def get_gamma(self) -> float:
        """Return the discount factor used by the agent."""
        return self._gamma

    def get_buffer_size(self) -> int:
        """Return the current number of transitions stored in the replay buffer."""
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """
        Add a single transition or a batch of transitions to the replay buffer.

        Parameters
        ----------
        board : np.ndarray
            Current state(s) in NHWC format (N, H, W, C) or (H, W, C).
        action : int or np.ndarray
            Action index or batch of actions.
        reward : np.ndarray
            Reward(s) for taking the action(s).
        next_board : np.ndarray
            Next state(s) in NHWC format.
        done : np.ndarray
            Episode termination flags.
        legal_moves : np.ndarray
            Legal moves mask(s) with shape (N, n_actions) or (n_actions,).
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    def train_agent(self, batch_size: int = 32, num_games: int = 1, reward_clip: bool = False):
        """
        Sample a batch from the replay buffer and perform one gradient update.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample from the replay buffer.
        num_games : int
            Kept for compatibility with original API (not used directly).
        reward_clip : bool
            If True, rewards are clipped; controlled via self._reward_clip.

        Returns
        -------
        float
            Scalar training loss from the last optimization step.
        """
        if self._buffer.get_current_size() < batch_size:
            # Not enough experience yet, reuse previous loss value.
            return float(self._last_loss)

        s, a_onehot, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        if self._reward_clip:
            # Optional value clipping to stabilise learning.
            r = np.clip(r, -1, 1)

        # Convert numpy arrays to tensors on the configured device.
        s_t = self._to_tensor(s)          # (N, C, H, W)
        next_s_t = self._to_tensor(next_s)
        r_t = torch.from_numpy(r.astype(np.float32)).to(self._device)          # (N, 1)
        done_t = torch.from_numpy(done.astype(np.float32)).to(self._device)    # (N, 1)
        a_idx = torch.from_numpy(
            np.argmax(a_onehot, axis=1).astype(np.int64)
        ).to(self._device)  # (N,)

        # Q(s, a) for actions actually taken.
        q_all = self._model(s_t)                           # (N, n_actions)
        q_sa = q_all.gather(1, a_idx.view(-1, 1)).squeeze(1)  # (N,)

        # Compute max_a' Q_target(s', a') using either target net or online net.
        with torch.no_grad():
            if self._use_target_net and self._target_net is not None:
                q_next_all = self._target_net(next_s_t)
            else:
                q_next_all = self._model(next_s_t)
            q_next_max = q_next_all.max(dim=1).values  # (N,)

        # Standard DQN target: r + gamma * (1 - done) * max_a' Q(s', a').
        target = r_t.view(-1) + self._gamma * (1.0 - done_t.view(-1)) * q_next_max

        loss = self._criterion(q_sa, target)

        # Backpropagation and optimization step.
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
        self._optimizer.step()

        self._last_loss = float(loss.detach().cpu().item())
        return self._last_loss

    def update_target_net(self):
        """
        Perform a hard update from the online network to the target network.

        Called periodically during training to stabilise Q-value targets.
        """
        if self._use_target_net and self._target_net is not None:
            self._target_net.load_state_dict(self._model.state_dict())

    def save_model(self, file_path: str = "", iteration: Optional[int] = None):
        """
        Save the current model (and optional target network) as PyTorch .pt files.

        This uses a naming scheme compatible with the original project, but
        avoids mixing TensorFlow/Keras .h5 files with PyTorch checkpoints.
        """
        if iteration is None:
            iteration = 0
        assert isinstance(iteration, int), "iteration should be an integer"

        os.makedirs(file_path, exist_ok=True)

        # Use .pt extension to clearly indicate PyTorch state_dict format.
        main_path = f"{file_path}/model_{iteration:06d}.pt"
        tgt_path = f"{file_path}/model_{iteration:06d}_target.pt"

        torch.save(self._model.state_dict(), main_path)

        if self._use_target_net and self._target_net is not None:
            torch.save(self._target_net.state_dict(), tgt_path)

        print(f"[SAVE] Saved PyTorch model to {main_path}")

    def load_model(self, file_path: str = "", iteration: Optional[int] = None):
        """
        Load model weights from a .pt checkpoint.

        A simple header check is used to guard against accidentally loading
        old HDF5/Keras files. If such a file is detected, a clear error is
        raised instructing the user to delete legacy checkpoints.
        """
        if iteration is None:
            iteration = 0
        assert isinstance(iteration, int), "iteration should be an integer"

        main_path = f"{file_path}/model_{iteration:06d}.pt"
        tgt_path = f"{file_path}/model_{iteration:06d}_target.pt"

        if not os.path.exists(main_path):
            raise FileNotFoundError(
                f"[LOAD ERROR] Could not find model file: {main_path}\n"
                f"Expected a PyTorch '.pt' file. "
                f"If you still have .h5 files from TensorFlow, delete them."
            )

        # Header check to ensure this is not an HDF5/Keras file.
        with open(main_path, "rb") as f:
            header = f.read(8)
            # HDF5 files start with: b'\\x89HDF\\r\\n\\x1a\\n'
            if header.startswith(b"\x89HDF"):
                raise ValueError(
                    f"[LOAD ERROR] File {main_path} appears to be an HDF5/Keras file, not a PyTorch .pt file.\n"
                    f"Delete all old model_*.h5 files and retrain."
                )

        print(f"[LOAD] Loading PyTorch model from {main_path}")
        state = torch.load(main_path, map_location=self._device)
        self._model.load_state_dict(state)

        # Load target network if available (older checkpoints may not have one).
        if self._use_target_net and self._target_net is not None:
            if os.path.exists(tgt_path):
                tgt_state = torch.load(tgt_path, map_location=self._device)
                self._target_net.load_state_dict(tgt_state)
                print(f"[LOAD] Loaded target net from {tgt_path}")
            else:
                print(
                    f"[LOAD WARNING] No target net found at: {tgt_path} "
                    f"(this is OK for older checkpoints)"
                )

    def move(self, board, legal_moves, value=None):
        """
        Select the action with maximum Q-value among legal moves.

        Supports both single-state (H, W, C) and batched (N, H, W, C) inputs.

        Parameters
        ----------
        board : np.ndarray
            Current state in NHWC format.
        legal_moves : np.ndarray
            Mask of allowed actions; shape (n_actions,) or (N, n_actions).
        value : unused
            Kept for compatibility with the original agent API.

        Returns
        -------
        int or np.ndarray
            Selected action index for single state or array of indices for batch.
        """
        # Ensure numpy arrays.
        board_arr = np.asarray(board)
        legal_arr = np.asarray(legal_moves)

        # Add batch dimension when a single state is provided.
        if board_arr.ndim == 3:
            board_arr = board_arr[None, ...]  # (1, H, W, C)
        elif board_arr.ndim != 4:
            raise ValueError(
                f"board must have shape (H,W,C) or (N,H,W,C), got {board_arr.shape}"
            )

        if legal_arr.ndim == 1:
            legal_arr = legal_arr[None, :]    # (1, n_actions)
        elif legal_arr.ndim != 2:
            raise ValueError(
                f"legal_moves must have shape (n_actions,) or (N,n_actions), got {legal_arr.shape}"
            )

        # Align batch sizes between state and legal move arrays.
        n = min(board_arr.shape[0], legal_arr.shape[0])
        board_arr = board_arr[:n]
        legal_arr = legal_arr[:n]

        # Compute Q-values for all actions.
        q = self._get_model_outputs(board_arr)  # (N, n_actions)

        # Mask out illegal moves by assigning -inf to their Q-values.
        masked = np.where(legal_arr == 1, q, -np.inf)
        actions = np.argmax(masked, axis=1)

        # For the single-game case, return a scalar action.
        if actions.shape[0] == 1:
            return int(actions[0])
        return actions

    def _to_tensor(self, board_nhwc: np.ndarray) -> torch.Tensor:
        """
        Convert board states from NHWC numpy arrays to NCHW torch tensors.

        The input is normalised using board/4.0, consistent with the
        original implementation.
        """
        x = np.asarray(board_nhwc)

        # Ensure batch dimension.
        if x.ndim == 3:
            x = x[None, ...]
        if x.ndim != 4:
            raise ValueError(f"Expected board with 3 or 4 dims, got shape {x.shape}")

        # Normalise and transpose to NCHW format.
        x = x.astype(np.float32) / 4.0
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.from_numpy(x).to(self._device)

    def _get_model_outputs(self, board_nhwc: np.ndarray, model: Optional[nn.Module] = None) -> np.ndarray:
        """
        Compute Q-values for given board states and return them as numpy arrays.

        Parameters
        ----------
        board_nhwc : np.ndarray
            Board states in NHWC format, shape (H, W, C) or (N, H, W, C).
        model : nn.Module, optional
            Alternative network to query (e.g. target net). Defaults to the
            main online network.

        Returns
        -------
        np.ndarray
            Q-values with shape (N, n_actions).
        """
        if model is None:
            model = self._model
        x = np.asarray(board_nhwc)
        if x.ndim == 3:
            x = x[None, ...]
        if x.ndim != 4:
            raise ValueError(f"Expected board with 3 or 4 dims, got shape {x.shape}")
        with torch.no_grad():
            x_t = self._to_tensor(x)
            q = model(x_t)  # (N, n_actions)
            return q.detach().cpu().numpy()




