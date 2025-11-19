# visualize_policy_gif.py
# Utility script for generating GIFs of a trained DQN agent in the Snake environment.

import argparse
import json
from agent import DeepQLearningAgent
from game_environment import Snake
from utils import visualize_game_gif


def main():
    """
    Entry point for policy visualisation.

    This script loads a given model checkpoint for the specified configuration
    version, creates a Snake environment, and then generates one or more GIFs
    showing how the agent behaves from random initial conditions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v17.1")
    parser.add_argument(
        "--iteration",
        type=int,
        default=50000,
        help="Which model iteration to load (as in model_XXXXXX.pt).",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=5,
        help="How many GIFs to generate from the same checkpoint.",
    )
    args = parser.parse_args()

    version = args.version

    # Load configuration parameters for the requested version.
    with open(f"model_config/{version}.json", "r") as f:
        m = json.load(f)

    board_size = m["board_size"]
    frames = m["frames"]
    max_time_limit = m["max_time_limit"]
    obstacles = bool(m["obstacles"])

    # Create a single-game Snake environment for visualisation only.
    # The max_time_limit_vis is aligned with the original
    # game_visualization.py script.
    max_time_limit_vis = 398
    env = Snake(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit_vis,
        obstacles=obstacles,
        version=version,
    )
    n_actions = env.get_num_actions()

    # Instantiate the PyTorch DQN agent and load the requested checkpoint.
    agent = DeepQLearningAgent(
        board_size=board_size,
        frames=frames,
        n_actions=n_actions,
        buffer_size=10,
        version=version,
    )

    agent.load_model(file_path=f"models/{version}", iteration=args.iteration)

    # Generate several GIFs from the same checkpoint but with different
    # random initial conditions.
    for i in range(args.num_games):
        out_path = f"images/game_visual_{version}_{args.iteration}_14_ob_{i}.gif"
        print(f"Lager {out_path} ...")
        visualize_game_gif(env, agent, path=out_path, debug=False, fps=12)


if __name__ == "__main__":
    main()
