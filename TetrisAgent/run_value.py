import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from random import sample
import copy
import wandb
from tqdm import tqdm

class FeatureVector:
    """Observation wrapper that returns a feature vector as observation.

    **State representation**
        A feature vector can contain different features of the board, such as the height of the stack or the number of holes.
        In the literature, this is often referred to as a state representation and many different features can be used. A
        discussion about the state representation can be found in "Reinforcement learning (RL) is a paradigm within machine
        learning that has been applied to Tetris, demonstrating the effect of state representation on performance
        (Hendriks)."

    **Features**
        For this wrapper, the features from https://github.com/uvipen/Tetris-deep-Q-learning-pytorch have been
        adapted. These features are:

        - The height of the stack in each column (list: int for each column)
        - The maximum height of the stack (int)
        - The number of holes in the stack (int)
        - The bumpiness of the stack (int)

        More features can be added in the future or by introducing new wrappers.
    """

    def __init__(
        self,
        report_height=True,
        report_max_height=True,
        report_holes=True,
        report_bumpiness=True,
    ):
        """Initialize the FeatureVectorObservation wrapper.

        Args:
            env (Tetris): The environment.
            report_height (bool, optional): Report the height of the stack in each column. Defaults to True.
            report_max_height (bool, optional): Report the maximum height of the stack. Defaults to True.
            report_holes (bool, optional): Report the number of holes in the stack. Defaults to True.
            report_bumpiness (bool, optional): Report the bumpiness of the stack. Defaults to True.
        """
        self.report_height = report_height
        self.report_max_height = report_max_height
        self.report_holes = report_holes
        self.report_bumpiness = report_bumpiness

    def calc_height(self, board):
        """Calculate the height of the board.

        Args:
            board (np.ndarray): The board.

        Returns:
            np.ndarray: The height of the stack in each column.
        """
        # Find the lowest non-zero element in each column
        heights = board.shape[0] - np.argmax(
            board != 0, axis=0
        )  # measure top to bottom to avoid holes
        heights = np.where(
            np.all(board == 0, axis=0), 0, heights
        )  # empty columns should be 0 (not 20)
        return heights

    def calc_max_height(self, board):
        """Calculate the maximum height of the board.

        Args:
            board (np.ndarray): The board.

        Returns:
            int: The maximum height of the board.
        """
        # Find the maximum height across all columns
        return np.max(self.calc_height(board))

    def calc_bumpiness(self, board):
        """Calculate the bumpiness of the board.

        Bumpiness is the sum of the absolute differences between adjacent column heights.

        Args:
            board (np.ndarray): The board.

        Returns:
            int: The bumpiness of the board.
        """
        heights = self.calc_height(board)
        # Calculate differences between adjacent heights and sum their absolute values
        return np.sum(np.abs(np.diff(heights)))

    def calc_holes(self, board):
        """Calculate the number of holes in the stack.

        Args:
            board (np.ndarray): The board.

        Returns:
            int: The number of holes in the stack.
        """
        # Create a mask of non-zero elements
        filled = board != 0
        # Calculate cumulative sum of filled cells from top to bottom
        cumsum = np.cumsum(filled, axis=0)
        # Count cells that are empty but have filled cells above them
        return np.sum((board == 0) & (cumsum > 0))

    def observation(self, observation, padding= 4):
        """Observation wrapper that returns the feature vector as the observation.

        Args:
            observation (dict): The observation from the base environment.

        Returns:
            np.ndarray: The feature vector.
        """
        # Board
        board_obs = observation
        ## Padding = self.env.unwrapped.padding
        board_obs = board_obs[
            0 : -padding,
            padding : -padding,
        ]

        features = []

        if self.report_height or self.report_max_height:
            height_vector = self.calc_height(board_obs)
            if self.report_height:
                features += list(height_vector)
            if self.report_max_height:
                max_height = np.max(height_vector)
                features.append(max_height)

        if self.report_holes:
            holes = self.calc_holes(board_obs)
            features.append(holes)

        if self.report_bumpiness:
            bumpiness = self.calc_bumpiness(board_obs)
            features.append(bumpiness)

        features = np.array(features, dtype=np.uint8)
        return features
    

class V_function(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden=1):
        super().__init__()
        _layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            _layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])
        _layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc1 = nn.Sequential(*_layers)
    def forward(self, x):
        x = self.fc1(x)
        return x

def valid_indices(obs):
    """Return indices of observations that are legal placements."""
    indices = []  # contains all env.legal_actions_mask
    for i in range(len(obs)):
        if not np.all(obs[i] == 1):
            indices.append(i)
    return indices

def next_states(obs):
    """Extract feature vectors for all valid next states."""
    feature_class = FeatureVector()
    padding = 4  # env.env.unwrapped.padding
    indices = valid_indices(obs)
    next_states_list = []
    for index in indices:
        feature = feature_class.observation(obs[index], padding)
        next_states_list.append(feature)
    return next_states_list

def curr_state(board):
    """Extract feature vector from current board state."""
    feature_class = FeatureVector()
    padding = 4  # env.env.unwrapped.padding
    return feature_class.observation(board, padding)

def soft_update(target, source, tau):
    """Perform a soft update of the target network parameters.
    
    Args:
        target (nn.Module): The target network.
        source (nn.Module): The source network.
        tau (float): Interpolation parameter.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def train(model: V_function, args):
    """Train the value function using DQN-style learning."""
    replay_buffer = deque(maxlen=100000)
    v_function = model
    optimizer = torch.optim.Adam(v_function.parameters(), lr=args["lr"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    # Target Value Network (using soft update)
    target_v_function = copy.deepcopy(v_function)
    target_v_function.eval()
    # Environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    env = GroupedActionsObservations(env, observation_wrappers=None, terminate_on_illegal_action=True)
    total_loss = 0
    total_reward = 0
    for epoch in tqdm(range(args["num_epochs"]), desc="Training"):
        if epoch < args["epsilon_decay_start_epoch"]:
            epsilon = args["init_epsilon"]
        elif args["epsilon_decay_start_epoch"] <= epoch < args["epsilon_decay_end_epoch"]:
            epsilon = args["final_epsilon"] + max(args["epsilon_decay_end_epoch"] - epoch, 0) * (args["init_epsilon"] - args["final_epsilon"]) / (args["epsilon_decay_end_epoch"] - args["epsilon_decay_start_epoch"])
        else:
            epsilon = args["final_epsilon"]
        observation = env.reset(seed=42)[0]
        terminated = False

        total_loss = 0.
        total_reward = 0.

        while not terminated:
            current_state = curr_state(env.env.unwrapped.board)  # Features of current state
            indices = valid_indices(observation)
            available_next_states = next_states(observation)
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                chosen_index = np.random.choice(np.arange(len(indices)))
                action = indices[chosen_index]
            else:
                with torch.no_grad():
                    values = v_function(torch.tensor(available_next_states).float().to(device)).squeeze(1)
                chosen_index = torch.argmax(values).cpu().numpy()
                action = indices[chosen_index]

            next_state = available_next_states[chosen_index]  # Features of obtained next state
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            observation = next_observation

            replay_buffer.append((current_state, reward, next_state, terminated))
            batch = sample(replay_buffer, min(args["batch_size"], len(replay_buffer)))
            current_states_batch, rewards_batch, next_states_batch, terminated_batch = zip(*batch)
            current_states_batch = torch.tensor(current_states_batch).float().to(device)
            next_states_batch = torch.tensor(next_states_batch).float().to(device)
            rewards_batch = torch.tensor(rewards_batch).float().to(device)
            with torch.no_grad():
                target_batch = rewards_batch + args["gamma"] * (1 - torch.tensor(terminated_batch).long().to(device)) * target_v_function(next_states_batch).squeeze(1)
            optimizer.zero_grad()
            
            loss = criterion(v_function(current_states_batch).squeeze(1), target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Soft update the target network
            soft_update(target_v_function, v_function, args["tau"])
        
        metrics = {
            "epoch": epoch,
            "train/total_reward": total_reward,
            "train/epsilon": epsilon,
            "train/loss": total_loss,
            "train/replay_buffer_size": len(replay_buffer),
        }

        if epoch % args["eval_freq"] == 0:
            eval_metrics, videos = evaluate(model, env, args)
            eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
            metrics.update(eval_metrics)
            v_function.train()
            target_v_function.train()
            if videos:
                for idx, video in enumerate(videos):
                    wandb_vid = wandb.Video(np.stack(video).transpose(0, 3, 1, 2), fps=10, format="mp4")
                    metrics[f"videos/{idx}"] = wandb_vid

        wandb.log(metrics)

def evaluate(model: V_function, env, args):
    """
    Evaluate the given model over a fixed number of episodes.
    """
    total_rewards = []
    n_steps = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    video_frames = []  # will hold frames from the first evaluation episode
    for i in range(args["n_eval_trajs"]):
        observation = env.reset()[0]
        terminated = False
        episode_reward = 0
        steps = 0
        # Capture frames for video logging
        frames = []
        while not terminated:
            frame = env.render()  # Capture the current frame
            frames.append(frame)
            current_state = curr_state(env.env.unwrapped.board)
            indices = valid_indices(observation)
            available_next_states = next_states(observation)
            with torch.no_grad():
                values = model(torch.tensor(available_next_states).float().to(device)).squeeze(1)
            chosen_index = torch.argmax(values).cpu().numpy()
            action = indices[chosen_index]

            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            observation = next_observation
            steps += 1

        total_rewards.append(episode_reward)
        n_steps.append(steps)
        video_frames.append(frames)
    avg_reward = np.mean(total_rewards)
    metrics = {
        "average_reward": avg_reward,
        "median_reward": np.median(total_rewards),
        "std_reward": np.std(total_rewards),
        "average_steps": np.mean(n_steps),
        "median_steps": np.median(n_steps),
    }

    return metrics, video_frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--init_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--n_eval_trajs", type=int, default=5)
    parser.add_argument("--epsilon_decay_start_epoch", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=13)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=1)
    args = parser.parse_args()
    args = vars(args)
    args["epsilon_decay_end_epoch"] = (args["num_epochs"] * 2) // 3
    
    model = V_function(args["input_dim"], args["hidden_dim"], args["output_dim"], n_hidden=args["n_hidden"])
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    wandb.init(project="tetris_rl", entity="AFIL", config=args)
    wandb.watch(model, log="all")
    
    model.train()
    train(model, args)

