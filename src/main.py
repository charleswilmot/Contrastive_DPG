from experiment import Experiment
from agent import Agent
from jax import random
import os
import re
import datetime
from tensorboardX import SummaryWriter
import numpy as np
import jax


if __name__ == '__main__':
    n_actors = 50
    discount_factor = 0.9
    noise_magnitude_limit = 0.5
    contrastive_loss_coef = 0.0001
    actor_learning_rate = 2e-5
    critic_learning_rate = 1e-3
    action_dim = 7
    n_sim = 16
    batch_size = 4
    exploration_prob = 0.4
    episode_length = 100
    lookback = 4
    smoothing = 0.2

    n_data_points = episode_length * n_sim * 8
    n_data_collect = n_data_points // (episode_length * n_sim)
    CUTOFF = min(n_data_collect * n_sim, 100)

    log_path = '../experiments'
    ids = [int(match.group(1)) for x in os.listdir(log_path) if (match := re.match('([0-9]+)_[a-zA-Z]+[0-9]+_[0-9]+-[0-9]+', x))]
    if ids:
        exp_id = 1 + max(ids)
    else:
        exp_id = 0
    run_name = f'{exp_id:03d}_{datetime.datetime.now():%b%d_%H-%M}'
    tensorboard_before = SummaryWriter(logdir=f'{log_path}/{run_name}/before', flush_secs=30)
    tensorboard_after_critic = SummaryWriter(logdir=f'{log_path}/{run_name}/after_critic', flush_secs=30)
    tensorboard_after_actor = SummaryWriter(logdir=f'{log_path}/{run_name}/after_actor', flush_secs=30)
    tensorboard_training = SummaryWriter(logdir=f'{log_path}/{run_name}/training', flush_secs=30)
    profiling_path = f'{log_path}/{run_name}/profiling'


    subkey = random.PRNGKey(0)
    original_key = subkey

    agent = Agent(
        n_actors,
        discount_factor,
        noise_magnitude_limit,
        contrastive_loss_coef,
        actor_learning_rate,
        critic_learning_rate,
        action_dim,
    )
    args = [n_sim, batch_size, exploration_prob, episode_length, agent]
    with Experiment(*args) as experiment:
        # experiment.restore('../checkpoints/init_weights.ckpt')
        data_buffer = np.zeros(shape=(n_sim * n_data_collect * lookback, episode_length), dtype=experiment._dtype)
        data = experiment.collect_episode_data_multi(n_data_collect * (lookback - 1), exploration_prob, subkey)
        data_buffer[n_sim * n_data_collect:] = data
        for i in range(100):
            key, subkey = random.split(subkey)
            if i == 1 and profile: jax.profiler.start_trace(profiling_path)
            data = experiment.collect_episode_data_multi(n_data_collect, exploration_prob, subkey)
            if i == 1 and profile: jax.profiler.stop_trace()

            start = (i % lookback) * n_sim * n_data_collect
            stop = ((i % lookback) + 1) * n_sim * n_data_collect
            data_buffer[start:stop] = data
            experiment.log_data(tensorboard_before, data[:CUTOFF], i)

            if i == 1 and profile: jax.profiler.start_trace(profiling_path)
            experiment.full_critic_training(tensorboard_training, data_buffer, 400, subkey, i)
            if i == 1 and profile: jax.profiler.stop_trace()

            experiment.log_data(tensorboard_after_critic, data[:CUTOFF], i)

            if i == 1 and profile: jax.profiler.start_trace(profiling_path)
            experiment.full_actor_training(tensorboard_training, data_buffer, 100, subkey, i)
            if i == 1 and profile: jax.profiler.stop_trace()

            experiment.log_data(tensorboard_after_actor, data[:CUTOFF], i)
            experiment.checkpoint(f'../checkpoints/{run_name}.ckpt')
            if not (i % 10) and i != 0:
                experiment.log_videos(tensorboard_after_actor, 8, exploration_prob, smoothing, original_key, i)
