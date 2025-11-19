# Utility functions for training, evaluation, visualisation and logging.
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import time
import pandas as pd
import sys
import numpy as np  # ensure numpy is imported as np


def calculate_discounted_rewards(rewards, discount_factor=0.99):
    """
    Compute discounted returns for a sequence of step rewards.

    This helper is primarily used for policy gradient and A2C methods
    where future rewards are folded backwards into a single return
    per time step.

    Parameters
    ----------
    rewards : np.ndarray
        1D array of rewards for a single episode.
    discount_factor : float, optional
        Discount factor gamma; should be < 1 for convergence.

    Returns
    -------
    np.ndarray
        Array of discounted rewards of the same shape as the input.
    """
    discounted_rewards = np.zeros(rewards.shape, dtype=np.int16)
    discounted_rewards[rewards.shape[0] - 1] = rewards[rewards.shape[0] - 1]
    i = rewards.shape[0] - 2
    while i > -1:
        discounted_rewards[i] = rewards[i] + discount_factor * discounted_rewards[i + 1]
        i -= 1
    return discounted_rewards.copy()


def play_game(env, agent, n_actions, n_games=100, epsilon=0.01, record=True,
              verbose=False, reset_seed=False, sample_actions=False,
              reward_type='current'):
    """
    Run a number of games sequentially using a single environment instance.

    This is the original, non-vectorised game loop and is used mainly for
    policy-gradient style training and evaluation with detailed reward
    handling.

    Parameters
    ----------
    env : object
        Environment instance exposing reset(), step() and get_values().
    agent : object
        Agent implementing move(), add_to_buffer() and optionally
        get_action_proba().
    n_actions : int
        Number of discrete actions.
    n_games : int, optional
        Total number of episodes to run.
    epsilon : float, optional
        Epsilon for epsilon-greedy exploration. Values < 0 disable epsilon.
    record : bool, optional
        If True, transitions are added to the agent's replay buffer.
    verbose : bool, optional
        If True, show a progress bar over episodes.
    reset_seed : bool, optional
        If True, reset the numpy RNG for each episode (fixed board layouts).
    sample_actions : bool, optional
        If True, sample actions from a probability distribution instead of
        taking argmax.
    reward_type : str, optional
        'current' for step-wise rewards or 'discounted_future' for returns.

    Returns
    -------
    list
        Total reward per episode for n_games episodes.
    """
    rewards = []
    iterator = tqdm(range(n_games)) if verbose else range(n_games)
    for _ in iterator:
        if reset_seed:
            np.random.seed(429834)
        rewards.append(0)
        s = env.reset()
        done = 0
        # Buffers used if discounted returns are computed at the end.
        s_list, action_list, reward_list, next_s_list, done_list = [], [], [], [], []
        while not done:
            if np.random.random() <= epsilon:
                # Pure epsilon-random exploration.
                action = np.random.choice(list(range(n_actions)))
            else:
                if sample_actions:
                    # Sample actions from a probability distribution provided by the agent.
                    probs = agent.get_action_proba(s)
                    action = np.random.choice(n_actions, p=probs)
                else:
                    # Greedy action based on Q-values or policy.
                    action = agent.move(s, env.get_values())
            next_s, reward, done, info = env.step(action)

            if record and (info['termination_reason'] != 'time_up'):
                if reward_type == 'current':
                    agent.add_to_buffer(s, action, reward, next_s, done)
                elif reward_type == 'discounted_future':
                    # Store transitions; they will be added once discounted.
                    s_list.append(s.copy())
                    action_list.append(action)
                    reward_list.append(reward)
                    next_s_list.append(next_s.copy())
                    done_list.append(done)
                else:
                    assert reward_type in ['current', 'discounted_future'], \
                        'reward type not understood !'
            s = next_s.copy()
            rewards[-1] += reward

        # After an episode, optionally compute discounted returns and add to buffer.
        if record and reward_type == 'discounted_future':
            reward_list = calculate_discounted_rewards(reward_list, agent.get_gamma())
            for i in range(len(reward_list)):
                agent.add_to_buffer(
                    s_list[i],
                    action_list[i],
                    reward_list[i],
                    next_s_list[i],
                    done_list[i],
                )
    return rewards


def play_game2(env, agent, n_actions, n_games=100, epsilon=0.01, record=True,
               verbose=False, reset_seed=False, sample_actions=False,
               reward_type='current', frame_mode=False, total_frames=10,
               total_games=None, stateful=False, debug=False):
    """
    Run multiple snake games in parallel using a vectorised environment.

    This function is the main driver for training in the Numpy-based
    SnakeNumpy environment. It supports both:
    - stepping until all parallel games are finished, or
    - stopping after a specified total number of frames or completed games.

    The function can record transitions into the replay buffer either with
    per-step rewards or discounted returns.

    Parameters
    ----------
    env : object
        Vectorised environment (e.g. SnakeNumpy) exposing reset(), step(),
        get_legal_moves(), get_values(), and attributes such as games/n_games.
    agent : object
        Agent implementing move(), add_to_buffer(), get_gamma().
    n_actions : int
        Number of discrete actions.
    n_games : int, optional
        Number of parallel games used for statistics and frame accounting.
    epsilon : float, optional
        Epsilon value for epsilon-greedy exploration.
    record : bool, optional
        If True, transitions are recorded into the agent's buffer.
    verbose : bool, optional
        Unused here (kept for API compatibility).
    reset_seed : bool, optional
        If True, use a fixed random seed for reproducibility.
    sample_actions : bool, optional
        If True, sample from probabilities instead of greedy moves.
    reward_type : str, optional
        'current' or 'discounted_future' reward handling.
    frame_mode : bool, optional
        If True, override termination based on total_frames/total_games instead
        of waiting for all games to finish.
    total_frames : int, optional
        Upper limit on number of frames to simulate in frame_mode.
    total_games : int, optional
        Upper limit on number of completed games in frame_mode.
    stateful : bool, optional
        Kept for compatibility; not used in this implementation.
    debug : bool, optional
        If True, prints internal counters at each step.

    Returns
    -------
    tuple
        (total_rewards, total_lengths, total_games_finished)
        aggregated over all parallel games.
    """
    rewards = 0
    lengths = 0
    if reset_seed:
        np.random.seed(42)

    # Buffers for optional discounted-reward handling.
    s_list, action_list, reward_list, next_s_list, done_list, legal_moves_list = [], [], [], [], [], []

    # Initialise environment state; support both reset() and new_game()/get_state().
    try:
        s = env.reset()
    except Exception:
        if hasattr(env, "new_game"):
            env.new_game()
            s = env.get_state()
        else:
            raise

    # Determine number of parallel games from environment attributes.
    if hasattr(env, "games"):
        n_env_games = env.games
    elif hasattr(env, "n_games"):
        n_env_games = env.n_games
    elif hasattr(env, "num_games"):
        n_env_games = env.num_games
    else:
        n_env_games = n_games

    done = np.zeros((n_env_games,), dtype=bool)
    frames = 0
    games = 0

    # Loop until one of the termination criteria is met.
    # 1) non-frame mode: continue until all games are done
    # 2) frame_mode without total_games: stop after total_frames
    # 3) frame_mode with total_games: stop after specified number of finished games
    while (not frame_mode and not done.all()) or \
          (frame_mode and total_games is None and frames < total_frames) or \
          (frame_mode and total_games is not None and games < total_games):
        if debug:
            print(f"[DEBUG] frames={frames} games={games} done.sum={done.sum()}", flush=True)

        # Retrieve legal moves; ensure shape is compatible with the number of env games.
        legal_moves = env.get_legal_moves()
        n_legal = legal_moves.shape[0]

        if hasattr(env, "games"):
            n = env.games
        elif hasattr(env, "n_games"):
            n = env.n_games
        elif hasattr(env, "num_games"):
            n = env.num_games
        else:
            n = n_legal

        if np.random.random() <= epsilon:
            # Epsilon-greedy: random action only among legal moves.
            rand_matrix = np.random.random((n, n_actions))
            legal_trim = legal_moves[:n]
            masked = np.where(legal_trim > 0, rand_matrix, -np.inf)
            action = np.argmax(masked, axis=1)
        else:
            # Policy-based actions from the agent.
            if sample_actions:
                probs = agent.get_action_proba(s)
                probs = probs[:n]
                # Cumulative sum sampling per row.
                action = ((probs / probs.sum(axis=1, keepdims=True)).cumsum(axis=1)
                          < np.random.random((probs.shape[0], 1))).sum(axis=1)
                action[action == n_actions] = n_actions - 1
            else:
                s_trim = s[:n] if s.shape[0] > n else s
                legal_trim = legal_moves[:n]
                action = agent.move(s_trim, legal_trim, env.get_values())

        # Step all games in parallel.
        next_s, reward, done, info, next_legal_moves = env.step(action)

        if record:
            if reward_type == 'current':
                agent.add_to_buffer(s, action, reward, next_s, done, next_legal_moves)
            elif reward_type == 'discounted_future':
                s_list.append(s.copy())
                action_list.append(action)
                reward_list.append(reward)
                next_s_list.append(next_s.copy())
                done_list.append(done)
                legal_moves_list.append(next_legal_moves)
            else:
                assert reward_type in ['current', 'discounted_future'], 'reward type not understood!'
        s = next_s.copy()
        rewards += np.dot(done, info['cumul_rewards'])
        frames += n_games
        games += done.sum()

    # If using discounted rewards, fold them backwards and add to buffer now.
    if record and reward_type == 'discounted_future':
        reward_list = calculate_discounted_rewards(reward_list, agent.get_gamma())
        for i in range(len(reward_list)):
            agent.add_to_buffer(
                s_list[i],
                action_list[i],
                reward_list[i],
                next_s_list[i],
                done_list[i],
                legal_moves_list[i],
            )

    # Aggregate length and reward statistics over completed games, if available.
    if isinstance(info, dict) and 'length' in info and 'cumul_rewards' in info:
        lengths = np.dot(done, info['length'])
        rewards = np.dot(done, info['cumul_rewards'])

    return rewards, lengths, games


def visualize_game(env, agent, path='images/game_visual.png', debug=False,
                    animate=False, fps=10):
    """
    Visualise a single Snake game either as a static grid of frames
    or as an MP4 animation written to disk.

    This is the original visualisation based on matplotlib + ffmpeg and
    is mainly used for inspecting how a trained agent behaves.
    """
    print('Starting Visualization')
    game_images = []
    qvalues = []
    food_count = []
    color_map = {0: 'lightgray', 1: 'g', 2: 'lightgreen', 3: 'r', 4: 'darkgray'}

    s = env.reset()
    board_size = env.get_board_size()
    game_images.append([s[:, :, 0], 0])
    done = 0
    while not done:
        legal_moves = env.get_legal_moves()
        a = agent.move(s, legal_moves, env.get_values())
        next_s, r, done, info, _ = env.step(a)
        qvalues.append(agent._get_model_outputs(s)[0])
        food_count.append(info['food'])
        game_images.append([next_s[:, :, 0], info['time']])
        s = next_s.copy()
        if debug:
            print(info['time'], qvalues[-1], a, r, info['food'], done, legal_moves)

    qvalues.append([0] * env.get_num_actions())
    food_count.append(food_count[-1])
    print('Game ran for {:d} frames'.format(len(game_images)))

    # Add a few static frames at the end to create a short pause.
    for _ in range(5):
        qvalues.append(qvalues[-1])
        food_count.append(food_count[-1])
        game_images.append(game_images[-1])

    # Animation mode (MP4) vs. grid of subplots.
    if animate:
        fig, axs = plt.subplots(
            1,
            1,
            figsize=(board_size // 2 + 1, board_size // 2 + 1)
        )
        anim = animation.FuncAnimation(
            fig,
            anim_frames_func,
            frames=game_images,
            blit=False,
            interval=10,
            repeat=True,
            init_func=None,
            fargs=(axs, color_map, food_count, qvalues),
        )
        anim.save(
            path,
            writer=animation.writers['ffmpeg'](
                fps=fps, metadata=dict(artist='Me'), bitrate=1800
            )
        )
    else:
        ncols = 5
        nrows = len(game_images) // ncols + (1 if len(game_images) % ncols > 0 else 0)
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(board_size * ncols, board_size * nrows),
            squeeze=False
        )
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < len(game_images):
                    axs[i, j] = anim_frames_func(
                        game_images[idx],
                        axs[i, j],
                        color_map,
                        food_count,
                        qvalues,
                    )
                else:
                    fig.delaxes(axs[i, j])
        fig.savefig(path, bbox_inches='tight')


# ----------------------------
# Animation helpers
# ----------------------------
def anim_init_func(axs):
    """Initialise an empty grid for animation (kept for compatibility)."""
    axs.clear
    return axs


def anim_frames_func(board_time, axs, color_map, food_count, qvalues):
    """
    Render a single frame of the game.

    Parameters
    ----------
    board_time : tuple
        (board, time_step) where board is a 2D array with integer codes.
    axs : matplotlib.axes.Axes
        Axes object to draw on.
    color_map : dict
        Mapping from board codes to colors.
    food_count : list
        Number of food items eaten at each time step.
    qvalues : list
        Q-value vectors per time step.

    Returns
    -------
    matplotlib.axes.Axes
        Updated axes with the current frame rendered.
    """
    axs.clear()
    board, time = board_time
    board_size = board.shape[0]
    half_width = 1.0 / (2 * board_size)
    delta = 0.025 * 2 * half_width
    half_width -= delta

    # Draw each grid cell as a colored rectangle.
    for i in range(board_size):
        for j in range(board_size):
            rect = Rectangle(
                ((half_width + delta) * (2 * j) + delta,
                 (half_width + delta) * (2 * (board_size - 1 - i)) + delta),
                width=2 * half_width,
                height=2 * half_width,
                color=color_map[board[i, j]]
            )
            axs.add_patch(rect)

    # Extract Q-values for this time step.
    raw_q = qvalues[time]
    if isinstance(raw_q, np.ndarray):
        if raw_q.ndim == 2:
            raw_q = raw_q[0]
        raw_q = raw_q.tolist()

    q1, q2, q3, q4 = [float(x) for x in raw_q]

    # Set title with time, score and Q-values.
    title = 'time:{:d}, score:{:d}\n{:.2f} {:.2f} {:.2f} {:.2f}'.format(
        time, int(food_count[time]), q1, q2, q3, q4
    )
    axs.set_title(title)
    plt.tight_layout()
    return axs


def plot_logs(data, title="Rewards and Loss Curve for Agent",
              loss_titles=['Loss']):
    """
    Plot learning curves (reward, length and one or more loss components).

    Parameters
    ----------
    data : str or dict
        Either a path to a CSV log file or a dictionary with pandas
        Series/arrays.
    title : str, optional
        Title for the plot.
    loss_titles : list of str, optional
        Titles for each loss component if multiple are logged.

    Example
    -------
    python -c "from utils import plot_logs; plot_logs('model_logs/v15.2.csv')"
    python -c "from utils import plot_logs; plot_logs('model_logs/v15.3.csv', loss_titles=['Total Loss', 'Actor Loss', 'Critic Loss'])"
    """
    loss_count = 1
    if isinstance(data, str):
        # Read from CSV file.
        data = pd.read_csv(data)
        if data['loss'].dtype == 'O':
            # Expand string-encoded list of losses into separate columns.
            loss_count = len(
                data.iloc[0, data.columns.tolist().index('loss')]
                .replace('[', '').replace(']', '').split(',')
            )
            for i in range(loss_count):
                data['loss_{:d}'.format(i)] = data['loss'].apply(
                    lambda x: float(
                        x.replace('[', '').replace(']', '').split(',')[i]
                    )
                )
            if len(loss_titles) != loss_count:
                loss_titles = loss_titles[0] * loss_count
    elif isinstance(data, dict):
        # Direct dictionary input is supported but not modified here.
        pass
    else:
        print('Provide a dictionary or file path for the data')

    fig, axs = plt.subplots(
        1 + loss_count + (1 if 'length_mean' in data.columns else 0),
        1,
        figsize=(8, 8)
    )
    axs[0].set_title(title)
    index = 0

    if 'length_mean' in data.columns:
        axs[0].plot(data['iteration'], data['length_mean'])
        axs[0].set_ylabel('Mean Length')
        index = 1

    axs[index].plot(data['iteration'], data['reward_mean'])
    axs[index].set_ylabel('Mean Reward')
    index += 1

    for i in range(index, index + loss_count):
        axs[i].plot(
            data['iteration'],
            data['loss_{:d}'.format(i - index) if loss_count > 1 else 'loss']
        )
        axs[i].set_ylabel(loss_titles[i - index])
        axs[i].set_xlabel('Iteration')

    plt.tight_layout()
    plt.show()


def visualize_game_gif(env, agent, path='images/game_visual.gif', debug=False, fps=12):
    """
    Run a single Snake game and save the result as an animated GIF.

    The visual style (colors, layout and title information) matches the
    original MP4-based visualisation, but uses PillowWriter instead of
    ffmpeg to produce a GIF.
    """
    print('Starting GIF Visualization')
    game_images = []
    qvalues = []
    food_count = []

    color_map = {0: 'lightgray', 1: 'g', 2: 'lightgreen', 3: 'r', 4: 'darkgray'}

    # Initialise environment and record the first frame.
    s = env.reset()
    board_size = env.get_board_size()
    game_images.append([s[:, :, 0], 0])
    done = 0

    # Roll out a full episode.
    while not done:
        legal_moves = env.get_legal_moves()
        a = agent.move(s, legal_moves, env.get_values())
        next_s, r, done, info, _ = env.step(a)

        # Track Q-values and food count for this state.
        qvalues.append(agent._get_model_outputs(s)[0])
        food_count.append(info['food'])

        game_images.append([next_s[:, :, 0], info['time']])
        s = next_s.copy()

        if debug:
            print(info['time'], qvalues[-1], a, r, info['food'], done, legal_moves)

    # Add a final dummy Q-vector and food count for the last frame.
    qvalues.append([0] * env.get_num_actions())
    food_count.append(food_count[-1])
    print('Game ran for {:d} frames'.format(len(game_images)))

    # Append a few static frames for a pause at the end.
    for _ in range(5):
        qvalues.append(qvalues[-1])
        food_count.append(food_count[-1])
        game_images.append(game_images[-1])

    # Build the GIF animation using PillowWriter (no external ffmpeg needed).
    fig, axs = plt.subplots(
        1, 1,
        figsize=(board_size // 2 + 1, board_size // 2 + 1)
    )

    anim = animation.FuncAnimation(
        fig,
        anim_frames_func,
        frames=game_images,
        blit=False,
        interval=10,
        repeat=True,
        fargs=(axs, color_map, food_count, qvalues),
    )

    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)
