from jax import random
import jax.numpy as jnp
import numpy as np
from simulation import SimulationPool
import logging
import sys
from distance_matrix import *
import pickle
import matplotlib
matplotlib.use('TkAgg') # necessary to avoid conflict with Coppelia's Qt
import matplotlib.pyplot as plt


def hann(n):
    X = jnp.linspace(0, jnp.pi, n)
    win = jnp.sin(X) ** 2
    return win / jnp.sum(win)


def compute_reward(registers_tm1, registers_t, goals):
    distance_tm1 = jnp.sum(jnp.float32(registers_tm1 != goals), axis=-1)
    distance_t = jnp.sum(jnp.float32(registers_t != goals), axis=-1)
    return distance_tm1 - distance_t


class Experiment:
    def __init__(self, n_sim, batch_size, exploration_prob, episode_length, agent):
        self._logger = logging.getLogger("Experiment")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info(f"initializing...")
        self._n_sim = n_sim
        self._batch_size = batch_size
        self._exploration_prob = exploration_prob
        self._simulations = SimulationPool(self._n_sim, scene='../3d_models/custom_timestep.ttt', guis=[])
        self._simulations.set_simulation_timestep(0.2)
        self._simulations.create_environment()
        self._simulations.set_control_loop_enabled(False)
        self._simulations.start_sim()
        self._simulations.step_sim()
        self._simulations.set_reset_poses()
        self._actions_dim = self._simulations.get_n_joints()[0]
        self._registers_dim = self._simulations.get_n_registers()[0]
        self._states_dim = 3 * self._actions_dim + self._registers_dim
        self._agent = agent
        key = random.PRNGKey(0)
        states, registers, goals = self.episode_reset(key)
        dummy_states = jnp.concatenate([states, goals], axis=-1)
        dummy_actions = jnp.zeros(shape=(dummy_states.shape[0], self._actions_dim))
        (
            self._actors_params,
            self._actors_learner_state,
            self._critic_params,
            self._critic_learner_state,
        ) = self._agent.init(dummy_states, dummy_actions, key)
        self._episode_length = episode_length
        self._dtype = np.dtype([
            ("states", np.float32, self._states_dim),
            ("goals", np.float32, self._registers_dim),
            ("actions", np.float32, self._actions_dim),
            ("rewards", np.float32),
        ])
        self._data_buffer = np.zeros(shape=(self._n_sim, self._episode_length), dtype=self._dtype)
        self._logger.info(f"initializing... done")

    def close(self):
        self._simulations.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def checkpoint(self, path):
        self._logger.info(f'Creating new checkpoint under {path}')
        ckpt = {
            "critic": self._critic_params,
            "actors": self._actors_params,
            "critic_learner_state": self._critic_learner_state,
            "actors_learner_state": self._actors_learner_state,
        }
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)

    def restore(self, path,
            critic=True,
            actors=True,
            critic_learner_state=None,
            actors_learner_state=None):
        self._logger.info(f'Restoring from checkpoint {path}')
        if critic and critic_learner_state is None:
            critic_learner_state = True
        if actors and actors_learner_state is None:
            actors_learner_state = True
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        if critic:
            self._critic_params = ckpt["critic"]
        if actors:
            self._actors_params = ckpt["actors"]
        if critic_learner_state:
            self._critic_learner_state = ckpt["critic_learner_state"]
        if actors_learner_state:
            self._actors_learner_state = ckpt["actors_learner_state"]

    def train_actor(self, states, goals):
        ######### self._logger.info(f'training actor {states.shape=} {goals.shape=}')
        self._actors_params, self._actors_learner_state, infos = self._agent.actor_learning_step(
            self._actors_learner_state,
            self._actors_params,
            self._critic_params,
            jnp.concatenate([states, goals], axis=-1),
        )
        return infos

    def train_critic(self, states, goals, actions, rewards):
        ######### self._logger.info(f'training critic {states.shape=} {goals.shape=} {actions.shape=} {rewards.shape=}')
        self._critic_params, self._critic_learner_state, infos = self._agent.critic_learning_step(
            self._critic_learner_state,
            self._actors_params,
            self._critic_params,
            jnp.concatenate([states, goals], axis=-1),
            actions,
            rewards,
        )
        return infos

    def collect_episode_data(self, exploration_prob, key):
        ######### self._logger.info(f'collecting episode data {exploration_prob=}')
        key, subkey = random.split(key)
        states, registers, goals = self.episode_reset(subkey)
        registers_tm1 = registers
        actions_tm1 = None
        actions_tm2 = None
        for iteration in range(self._episode_length):
            key, subkey = random.split(subkey)
            actions_exp = self._agent.get_explorative_action(
                self._actors_params,
                self._critic_params,
                jnp.concatenate([states, goals], axis=-1),
                subkey,
            )
            actions = self._agent.get_action(
                self._actors_params,
                self._critic_params,
                jnp.concatenate([states, goals], axis=-1),
                actions_tm2=actions_tm2,
                actions_tm1=actions_tm1,
            )
            explore = random.bernoulli(subkey, p=exploration_prob)
            if explore:
                actions_apply = actions_exp
            else:
                actions_apply = actions
            self._data_buffer[:, iteration]["states"] = states
            self._data_buffer[:, iteration]["goals"] = goals
            self._data_buffer[:, iteration]["actions"] = actions
            with self._simulations.distribute_args():
                ######### self._logger.debug(f'{explore=} {actions=}')
                states_registers = self._simulations.apply_action(actions)
            states = jnp.stack([s for s, r in states_registers])
            registers = jnp.stack([r for s, r in states_registers])
            rewards = compute_reward(registers_tm1, registers, goals)
            self._data_buffer[:, iteration]["rewards"] = rewards
            registers_tm1 = registers
            actions_tm2 = actions_tm1
            actions_tm1 = actions_apply
        return self._data_buffer

    def collect_episode_data_multi(self, n_data_collect, exploration_prob, key):
        data = np.zeros(shape=(self._n_sim * n_data_collect, self._episode_length), dtype=self._dtype)
        self._logger.info(f'collecting data (multi) {n_data_collect=}')
        key, subkey = random.split(key)
        for i in range(n_data_collect):
            self._logger.info(f'collecting data  --  {i+1}/{n_data_collect}')
            key, subkey = random.split(subkey)
            self.collect_episode_data(exploration_prob, subkey)
            data[i * self._n_sim:i * self._n_sim + self._n_sim] = self._data_buffer
        return data

    def episode_reset(self, key):
        key1, key2, key3 = random.split(key, num=3)
        with self._simulations.distribute_args() as n_sim:
            # n_sim = len(self._simulations._active_producers_indices)
            goals = random.bernoulli(key1, shape=(n_sim, self._registers_dim))
            registers = random.bernoulli(key2, shape=(n_sim, self._registers_dim))
            actions = random.uniform(key3, shape=(n_sim, self._actions_dim), minval=-1, maxval=1)
            states_registers = self._simulations.reset(registers, goals, actions)
        states = jnp.stack([s for s, r in states_registers])
        return states, registers.astype(jnp.int32), goals.astype(jnp.int32)

    def full_critic_training(self, tensorboard, data, n, key, iteration):
        self._logger.info(f'full critic training {data.shape=}')
        key, subkey = random.split(key)
        mean_critic_losses = []
        for i in range(n):
            self._logger.info(f'training  --  {i+1}/{n}')
            key, subkey = random.split(subkey)
            indices = random.choice(
                subkey,
                data.shape[0],
                shape=(self._batch_size,),
                replace=False,
            )
            batch = data[np.asarray(indices)]
            infos = self.train_critic(
                batch["states"],
                batch["goals"],
                batch["actions"],
                batch["rewards"],
            )
            mean_critic_losses.append(infos["mean_critic_loss"])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = jnp.array(mean_critic_losses)
        for size, alpha in [(4, 0.1), (20, 0.3), (100, 1.0)]:
            X = jnp.arange(size // 2, n - size // 2 + 1)
            Y = jnp.convolve(data, hann(size), mode='valid')
            ax.plot(X, Y, color='b', alpha=alpha)
        fig.tight_layout()
        tensorboard.add_figure('training/mean_critic_loss(smoothed)', fig, iteration, close=True)

    def full_actor_training(self, tensorboard, data, n, key, iteration):
        self._logger.info(f'full actor training {data.shape=}')
        key, subkey = random.split(key)
        mean_delta_actions_norm = []
        max_delta_actions_norm = []
        mean_actor_losses = []
        mean_smoothings = []
        for i in range(n):
            self._logger.info(f'training  --  {i+1}/{n}')
            key, subkey = random.split(subkey)
            indices = random.choice(
                subkey,
                data.shape[0],
                shape=(self._batch_size, ),
                replace=False,
            )
            batch = data[np.asarray(indices)]
            infos = self.train_actor(
                batch["states"],
                batch["goals"],
            )
            mean_delta_actions_norm.append(infos["mean(|delta_actions|)"])
            max_delta_actions_norm.append(infos["max(|delta_actions|)"])
            mean_actor_losses.append(infos["mean_actor_loss"])
            mean_smoothings.append(infos["mean(|d2actions_dt2|)"])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mean_delta_actions_norm)
        fig.tight_layout()
        tensorboard.add_figure('training/mean_delta_actions_norm', fig, iteration, close=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(max_delta_actions_norm)
        fig.tight_layout()
        tensorboard.add_figure('training/max_delta_actions_norm', fig, iteration, close=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mean_actor_losses)
        fig.tight_layout()
        tensorboard.add_figure('training/mean_actor_loss', fig, iteration, close=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mean_smoothings)
        fig.tight_layout()
        tensorboard.add_figure('training/mean_smoothing', fig, iteration, close=True)

    def log_data(self, tensorboard, data, iteration):
        self._agent.log_data(
                    self._actors_params,
                    self._critic_params,
                    jnp.concatenate([data["states"], data["goals"]], axis=-1),
                    data["actions"],
                    data["rewards"],
                    tensorboard,
                    iteration,
        )

    def get_videos(self, n, exploration_prob, key, width=200, height=200):
        self._logger.info(f'Creating a video {n=} {exploration_prob=} {width=} {height=}')
        # add cameras
        self._logger.debug('adding cameras')
        cam_ids = self._simulations.add_camera(
            position=(1.15, 1.35, 1),
            orientation=(
                24 * np.pi / 36,
                -7 * np.pi / 36,
                 4 * np.pi / 36
            ),
            resolution=(height, width),
        )
        # collect data
        todo = n
        videos = np.zeros(shape=(n, self._episode_length, width, height, 3))
        while todo > 0:
            self._logger.info(f'remaining episodes: {todo}/{n}')
            key, subkey = random.split(key)
            doing = min(self._n_sim, todo)
            actions_tm1 = None
            actions_tm2 = None
            with self._simulations.specific(list(range(doing))):
                # collect episodes
                states, registers, goals = self.episode_reset(subkey)
                for it in range(self._episode_length):
                    key, subkey = random.split(key)

                    actions_exp = self._agent.get_explorative_action(
                        self._actors_params,
                        self._critic_params,
                        jnp.concatenate([states, goals], axis=-1),
                        subkey,
                    )
                    actions = self._agent.get_action(
                        self._actors_params,
                        self._critic_params,
                        jnp.concatenate([states, goals], axis=-1),
                        actions_tm2=actions_tm2,
                        actions_tm1=actions_tm1,
                    )
                    explore = random.bernoulli(subkey, p=exploration_prob)
                    if explore:
                        actions_apply = actions_exp
                    else:
                        actions_apply = actions

                    with self._simulations.distribute_args():
                        states_registers = self._simulations.apply_action(actions_apply)
                    states = jnp.stack([s for s, r in states_registers])
                    with self._simulations.distribute_args():
                        frames = np.array(self._simulations.get_frame(cam_ids))
                    videos[todo - doing:todo, it] = frames

                    actions_tm2 = actions_tm1
                    actions_tm1 = actions_apply
            todo -= doing
        # delete cameras
        self._logger.debug('deleting cameras')
        with self._simulations.distribute_args():
            self._simulations.delete_camera(cam_ids)
        return videos

    def log_videos(self, tensorboard, n, exploration_prob, key, iteration, width=200, height=200):
        videos = self.get_videos(n, exploration_prob, key, width=width, height=height)
        tensorboard.add_video('videos/explorative', videos, iteration, fps=25, dataformats='NTHWC')
        videos = self.get_videos(n, 0.0, key, width=width, height=height)
        tensorboard.add_video('videos/non_explorative', videos, iteration, fps=25, dataformats='NTHWC')
