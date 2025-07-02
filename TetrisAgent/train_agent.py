import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
import os
from tetris_gymnasium.envs import Tetris


class GymnasiumToGymWrapper(gym.Wrapper):
    """Optional wrapper to merge Gymnasium's (terminated, truncated) flags for SB3 compatibility."""
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

@hydra.main(config_path="conf", config_name="dqn")
def main(cfg: DictConfig):
    print("Running with configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb with logging config; convert Hydra config to a standard dict.
    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.get("entity", None),
        config=OmegaConf.to_container(cfg.training, resolve=True)
    )

    # Create the training environment with Gymnasium.
    env = gym.make(cfg.training.env, render_mode=cfg.training.get("render_mode", None))
    env = Monitor(env)
    # Optionally record training video.
    if cfg.training.get("record_video", False):
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(
            env,
            video_folder=cfg.training.get("video_folder", "videos/"),
            episode_trigger=lambda episode_id: episode_id % cfg.training.get("video_trigger_episode", 10) == 0
        )
    # Optional: use GymnasiumToGymWrapper if needed for SB3 compatibility

    # Create a vectorized evaluation environment.
    eval_env = make_vec_env(
        cfg.training.env,
        n_envs=cfg.training.eval.n_eval_envs,
        seed=0,
    )

    # Compute evaluation frequency: when using multiple training envs, the agent is evaluated every
    # (eval_freq / n_training_envs) training steps.
    n_training_envs = cfg.training.get("n_training_envs", 1)
    eval_frequency = max(cfg.training.eval.eval_freq // n_training_envs, 1)

    # Create the evaluation callback.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.training.eval.best_model_save_path,
        log_path=cfg.training.eval.log_path,
        eval_freq=eval_frequency,
        n_eval_episodes=cfg.training.eval.n_eval_episodes,
        deterministic=cfg.training.eval.deterministic,
        render=cfg.training.eval.render,
    )

    # Combine the Wandb callback and the evaluation callback.
    combined_callback = CallbackList([
        WandbCallback(
            gradient_save_freq=100,  # Log gradients every 100 steps.
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        ),
        eval_callback
    ])

    # Initialize the DQN model.
    model = DQN(
        cfg.training.policy,
        env,
        learning_rate=cfg.training.learning_rate,
        buffer_size=cfg.training.buffer_size,
        batch_size=cfg.training.batch_size,
        gamma=cfg.training.gamma,
        exploration_fraction=cfg.training.exploration_fraction,
        exploration_final_eps=cfg.training.exploration_final_eps,
        train_freq=cfg.training.train_freq,
        target_update_interval=cfg.training.target_update_interval,
        verbose=1,
    )

    # Train the model with the combined callback.
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=combined_callback
    )

    # Upload recorded videos to wandb if enabled
    if cfg.training.get("record_video", False):
        video_folder = cfg.training.get("video_folder", "videos/")
        if os.path.exists(video_folder):
            for filename in os.listdir(video_folder):
                if filename.endswith(".mp4"):
                    video_path = os.path.join(video_folder, filename)
                    wandb.log({"evaluation_video": wandb.Video(video_path, caption="Evaluation Video")})

    # Save the final model.
    model.save("dqn_tetris")
    env.close()

if __name__ == "__main__":
    main()
