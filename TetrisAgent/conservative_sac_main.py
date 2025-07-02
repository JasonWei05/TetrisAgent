"""Conservative SAC Training Script

Trains a Conservative SAC agent using a transformer backbone.
"""
import os
import sys
import shutil
from copy import deepcopy

import numpy as np
import mlxu
import gym
import wandb

from conservative_sac import ConservativeSAC
from replay_buffer import get_d4rl_dataset, subsample_batch
from jax_utils import batch_to_jax
from model import TanhGaussianPolicy, FullyConnectedHead
from sampler import TrajSampler
from transformer_backbone import TransformerBackbone

sys.path.append('..')
from viskit.logging import logger, setup_logger


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    transformer_n_layer=1,
    transformer_hidden_size=int(768/4),
    a_vocab_size = 16,
    o_vocab_size = 16,

    n_epochs=700,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    save_model=False,
    save_interval=2,
    save_dir='checkpoints',
    restore_model=False,
    restore_path='',

    cql=ConservativeSAC.get_default_config(),
    logging=mlxu.WandBLogger.get_default_config(),
)


def main(argv):
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = mlxu.WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )
    wandb.run.log_code(".")

    mlxu.jax_utils.set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)

    dataset = get_d4rl_dataset(eval_sampler.env)
    rewards_max = np.max(dataset['rewards'])
    rewards_min = np.min(dataset['rewards'])
        
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)


    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]
    
    a_vocab_size = FLAGS.a_vocab_size
    o_vocab_size = FLAGS.o_vocab_size

    transformer_backbone_cfg = {
        'a_vocab_size': a_vocab_size,
        'o_vocab_size': o_vocab_size,
        'n_layer': FLAGS.transformer_n_layer,
        'n_head': 12,
        'hidden_size': FLAGS.transformer_hidden_size,
        'add_eos_token': False,
        'n_S': observation_dim,
        'n_A': action_dim,
    }
    
    transformer_backbone = TransformerBackbone(bt_config=transformer_backbone_cfg)

    turn_qf_head = FullyConnectedHead(
        observation_dim=FLAGS.transformer_hidden_size, 
        arch=FLAGS.qf_arch, 
        orthogonal_init=FLAGS.orthogonal_init
    )

    policy_head = TanhGaussianPolicy(
        FLAGS.transformer_hidden_size, 1, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, transformer_backbone, turn_qf_head, policy_head, observation_dim, action_dim)

    if FLAGS.restore_model:
        checkpoint = sac.restore_checkpoint(FLAGS.restore_path)
        epoch_counter = range(FLAGS.n_epochs)
        best_eval_return = checkpoint['best_eval_return']
    else:
        epoch_counter = range(FLAGS.n_epochs)
        best_eval_return = -np.inf

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.env)
    # Delete all previous checkpoints
    if FLAGS.save_model:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    

    viskit_metrics = {}
    for epoch in epoch_counter:
        metrics = {'epoch': epoch}
        with mlxu.Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                metrics.update(mlxu.prefix_metrics(
                    sac.train(batch, epoch=epoch, bc=epoch < FLAGS.bc_epochs), 'sac'
                ))

        with mlxu.Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sac.policy_func,
                    FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )

        # Save the best model
        if FLAGS.save_model:
            save_kwargs = dict(
                epoch=epoch,
                best_eval_return=best_eval_return,
                variant=variant,
            )
            if 'average_return' in metrics:
                if metrics['average_return'] > best_eval_return:
                    best_eval_return = metrics['average_return']
                    sac.save_checkpoint(save_dir, filename='best', **save_kwargs)

                    for file in os.listdir(save_dir):
                        if file.endswith('.result'):
                            os.remove(os.path.join(save_dir, file))
                    result_path = os.path.join(
                        save_dir, 
                        f'best_return_{metrics["average_normalizd_return"]:.2f}.result'
                    )
                    open(result_path, "a").close()

            if epoch % FLAGS.save_interval == 0:
                sac.save_checkpoint(save_dir, filename=f'epoch_{epoch}', **save_kwargs)
        
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    
    # Uncomment to save final model
    # if FLAGS.save_model:
    #     save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
    #     wandb_logger.save_pickle(save_data, 'model.pkl')
    
if __name__ == '__main__':
    mlxu.run(main)
