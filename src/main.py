import matplotlib
matplotlib.use('Agg') # necessary to avoid conflict with Coppelia's Qt
from experiment import Experiment
from exploration import ExplorationConfig
from agent import Agent
from jax import random
import os
import re
import datetime
from tensorboardX import SummaryWriter
import numpy as np
import jax


if __name__ == '__main__':
    # https://en.wikipedia.org/wiki/Kissing_number

    discount_factor = 0.9
    noise_magnitude_limit = 0.5
    hierarchization_coef = 1.0 # has been increased from 0.01 to 0.1, then from 0.01 to 1.0

    k = 1.15
    SQRT2 = 1.41421356237
    SAFETY = 2
    minmax_factor = 1.5
    dmin2 = 0.6
    dmax2 = dmin2 * minmax_factor
    dmin1 = SAFETY * SQRT2 * (dmax2)
    dmax1 = dmin1 * minmax_factor
    dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
    dmax0 = dmin0 * minmax_factor

    hierarchization_config = (
        (45, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (4, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    # level=0 - dim=60 - d_min=2.54 - d_max=3.81 - slope_min=0.86 - slope_max=0.86
    # level=1 - dim=3 - d_min=0.6 - d_max=0.90 - slope_min=0.75 - slope_max=0.75

    actor_learning_rate = 2e-5 # used to be 2e-5
    critic_learning_rate = 1e-3
    action_dim = 7
    n_sim = 20
    batch_size = 4
    exploration_config = ExplorationConfig(
        type="exploration_prob",
        N=4000,
        interpolation_type='cosine',
        upsilon_t0=0.2,
        upsilon_tN=0.6,
        exploration_prob_t0=0.9,
        exploration_prob_tN=0.1,
        softmax_temperature_t0=1.0,
        softmax_temperature_tN=0.25,
    )
    episode_length = 100
    lookback = 4
    smoothing = 0.0  # 0.04
    PRNGKey_start = 0
    n_expl_ep_per_it = 80
    n_nonexpl_ep_per_it = 80
    experiment_length_in_ep = 16000
    n_critic_training_per_loop_iteration = 400
    n_actor_training_per_loop_iteration = 100
    tensorboard_log = True
    restore_path = None


    log_path = '../experiments'
    os.makedirs(log_path, exist_ok=True)
    checkpoints_path = '../checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    ids = [int(match.group(1)) for x in os.listdir(log_path) if (match := re.match('([0-9]+)_[a-zA-Z]+[0-9]+_[0-9]+-[0-9]+', x))]
    if ids:
        exp_id = 1 + max(ids)
    else:
        exp_id = 0
    run_name = f'{exp_id:03d}_{datetime.datetime.now():%b%d_%H-%M}'

    agent = Agent(
        discount_factor,
        noise_magnitude_limit,
        hierarchization_config,
        hierarchization_coef,
        actor_learning_rate,
        critic_learning_rate,
        action_dim,
    )

    args = [n_sim, batch_size, smoothing, episode_length, agent]
    with Experiment(*args) as experiment:
        experiment.mainloop(
            PRNGKey_start=PRNGKey_start,
            lookback=lookback,
            n_expl_ep_per_it=n_expl_ep_per_it,
            n_nonexpl_ep_per_it=n_nonexpl_ep_per_it,
            experiment_length_in_ep=experiment_length_in_ep,
            n_critic_training_per_loop_iteration=n_critic_training_per_loop_iteration,
            n_actor_training_per_loop_iteration=n_actor_training_per_loop_iteration,
            exploration_config=exploration_config,
            tensorboard_log=tensorboard_log,
            restore_path=restore_path,
            path=f'{log_path}/{run_name}',
        )
