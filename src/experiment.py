from jax import random
import jax.numpy as jnp
import numpy as np
from simulation import SimulationPool
import logging
import sys
from distance_matrix import *
import pickle
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


def hann(n):
    X = jnp.linspace(0, jnp.pi, n)
    win = jnp.sin(X) ** 2
    return win / jnp.sum(win)


def compute_reward(registers_tm1, registers_t, goals):
    distance_tm1 = jnp.sum(jnp.float32(registers_tm1 != goals), axis=-1)
    distance_t = jnp.sum(jnp.float32(registers_t != goals), axis=-1)
    return distance_tm1 - distance_t


class Experiment:
    def __init__(self, n_sim, batch_size, smoothing, episode_length, agent):
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
        self._smoothing = smoothing
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
            self._actor_params,
            self._actor_learner_state,
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
            "actor": self._actor_params,
            "critic_learner_state": self._critic_learner_state,
            "actor_learner_state": self._actor_learner_state,
        }
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)

    def restore(self, path,
            critic=True,
            actor=True,
            critic_learner_state=None,
            actor_learner_state=None):
        self._logger.info(f'Restoring from checkpoint {path}')
        if critic and critic_learner_state is None:
            critic_learner_state = True
        if actor and actor_learner_state is None:
            actor_learner_state = True
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        if critic:
            self._critic_params = ckpt["critic"]
        if actor:
            self._actor_params = ckpt["actor"]
        if critic_learner_state:
            self._critic_learner_state = ckpt["critic_learner_state"]
        if actor_learner_state:
            self._actor_learner_state = ckpt["actor_learner_state"]

    def train_actor(self, states, goals):
        ######### self._logger.info(f'training actor {states.shape=} {goals.shape=}')
        self._actor_params, self._actor_learner_state, infos = self._agent.actor_learning_step(
            self._actor_learner_state,
            self._actor_params,
            self._critic_params,
            jnp.concatenate([states, goals], axis=-1),
        )
        return infos

    def train_critic(self, states, goals, actions, rewards):
        ######### self._logger.info(f'training critic {states.shape=} {goals.shape=} {actions.shape=} {rewards.shape=}')
        self._critic_params, self._critic_learner_state, infos = self._agent.critic_learning_step(
            self._critic_learner_state,
            self._actor_params,
            self._critic_params,
            jnp.concatenate([states, goals], axis=-1),
            actions,
            rewards,
        )
        return infos

    def collect_episode_data(self, key, exploration, episode_length=None, smoothing=None):
        if episode_length is None:
            episode_length = self._episode_length
        if smoothing is None:
            smoothing = self._smoothing
        data = np.zeros(shape=(self._n_sim, episode_length), dtype=self._dtype)
        ######### self._logger.debug(f'collecting episode data {upsilon=}')
        key, subkey = random.split(key)
        states, registers, goals = self.episode_reset(subkey)
        registers_tm1 = registers
        actions_tm1 = None
        actions_tm2 = None
        for iteration in range(episode_length):
            key, subkey = random.split(subkey)
            actions = self._agent.get_explorative_action(
                self._actor_params,
                self._critic_params,
                jnp.concatenate([states, goals], axis=-1),
                subkey,
                exploration,
                actions_tm2=actions_tm2,
                actions_tm1=actions_tm1,
                smoothing=smoothing,
            )
            data[:, iteration]["states"] = states
            data[:, iteration]["goals"] = goals
            data[:, iteration]["actions"] = actions
            with self._simulations.distribute_args():
                ######### self._logger.debug(f'{explore=} {actions=}')
                actions_tuple = tuple(a for a in actions)
                states_registers = self._simulations.apply_action(actions_tuple)
            states = jnp.stack([s for s, r in states_registers])
            registers = jnp.stack([r for s, r in states_registers])
            rewards = compute_reward(registers_tm1, registers, goals)
            data[:, iteration]["rewards"] = rewards
            registers_tm1 = registers
            actions_tm2 = actions_tm1
            actions_tm1 = actions
        return data

    def collect_episode_data_multi(self, n_data_collect, key, exploration, episode_length=None, smoothing=None):
        if episode_length is None:
            episode_length = self._episode_length
        if smoothing is None:
            smoothing = self._smoothing
        data = np.zeros(shape=(self._n_sim * n_data_collect, episode_length), dtype=self._dtype)
        self._logger.info(f'collecting data (multi) {n_data_collect=}')
        key, subkey = random.split(key)
        for i in range(n_data_collect):
            self._logger.info(f'collecting data  --  {i+1}/{n_data_collect}')
            key, subkey = random.split(subkey)
            data[i * self._n_sim:i * self._n_sim + self._n_sim] = \
                self.collect_episode_data(subkey, exploration, episode_length, smoothing)
        return data

    def episode_reset(self, key):
        key1, key2, key3 = random.split(key, num=3)
        with self._simulations.distribute_args() as n_sim:
            goals = tuple(random.bernoulli(subkey, shape=(self._registers_dim,)) for subkey in random.split(key1, num=n_sim))
            registers = tuple(random.bernoulli(subkey, shape=(self._registers_dim,)) for subkey in random.split(key2, num=n_sim))
            actions = tuple(random.uniform(subkey, shape=(self._actions_dim,), minval=-1, maxval=1) for subkey in random.split(key3, num=n_sim))
            states_registers = self._simulations.reset(registers, goals, actions)
        states = jnp.stack([s for s, r in states_registers])
        return states, jnp.array(registers).astype(jnp.int32), jnp.array(goals).astype(jnp.int32)

    def full_critic_training(self, data, n, key, iteration, tensorboard=None):
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
        if tensorboard is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            data = jnp.array(mean_critic_losses)
            for size, alpha in [(4, 0.1), (20, 0.3), (100, 1.0)]:
                X = jnp.arange(size // 2, n - size // 2 + 1)
                Y = jnp.convolve(data, hann(size), mode='valid')
                ax.plot(X, Y, color='b', alpha=alpha)
            fig.tight_layout()
            tensorboard.add_figure('training/mean_critic_loss(smoothed)', fig, iteration, close=True)

    def full_actor_training(self, data, n, key, iteration, tensorboard=None):
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

        if tensorboard is not None:
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

    def log_data(self, tensorboard, data, iteration, training_return, testing_return, exploration):
        self._agent.log_data(
                    self._actor_params,
                    self._critic_params,
                    jnp.concatenate([data["states"], data["goals"]], axis=-1),
                    data["actions"],
                    data["rewards"],
                    tensorboard,
                    iteration,
                    training_return,
                    testing_return,
                    exploration,
        )

    def get_videos(self, n, key, exploration, episode_length=None, smoothing=None, width=200, height=200):
        if episode_length is None:
            episode_length = self._episode_length
        if smoothing is None:
            smoothing = self._smoothing
        self._logger.info(f'Creating a video {n=} {upsilon=} {width=} {height=}')
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
                    key, subkey = random.split(subkey)
                    actions = self._agent.get_explorative_action(
                        self._actor_params,
                        self._critic_params,
                        jnp.concatenate([states, goals], axis=-1),
                        subkey,
                        exploration,
                        actions_tm2=actions_tm2,
                        actions_tm1=actions_tm1,
                        smoothing=smoothing,
                    )
                    with self._simulations.distribute_args():
                        actions_tuple = tuple(a for a in actions)
                        states_registers = self._simulations.apply_action(actions_tuple)
                    states = jnp.stack([s for s, r in states_registers])
                    with self._simulations.distribute_args():
                        frames = np.array(self._simulations.get_frame(cam_ids))
                    videos[todo - doing:todo, it] = frames

                    actions_tm2 = actions_tm1
                    actions_tm1 = actions
            todo -= doing
        # delete cameras
        self._logger.debug('deleting cameras')
        with self._simulations.distribute_args():
            self._simulations.delete_camera(cam_ids)
        return videos

    def log_videos(self, name, tensorboard, n, key, exploration, iteration, episode_length=None, smoothing=None, width=200, height=200):
        if episode_length is None:
            episode_length = self._episode_length
        if smoothing is None:
            smoothing = self._smoothing
        videos = self.get_videos(n, smoothing, key, exploration, width=width, height=height)
        tensorboard.add_video(f'videos/{name}', videos, iteration, fps=25, dataformats='NTHWC')

    def mainloop(self, PRNGKey_start, lookback, n_episodes_per_loop_iteration,
        experiment_length_in_ep, n_critic_training_per_loop_iteration,
        n_actor_training_per_loop_iteration, exploration_config, tensorboard_log,
        restore_path, path, database=None, experiment_id=None):

        if n_episodes_per_loop_iteration % self._n_sim != 0:
            raise RuntimeError(f"{n_episodes_per_loop_iteration=} not divisible by {self._n_sim=}")
        if experiment_length_in_ep % n_episodes_per_loop_iteration != 0:
            raise RuntimeError(f"{n_episodes_per_loop_iteration=} not divisible by {self._n_sim=}")

        if tensorboard_log:
            tensorboard = SummaryWriter(logdir=path)
        else:
            tensorboard = None

        subkey = random.PRNGKey(PRNGKey_start)
        original_key = subkey
        n_data_collect = int(n_episodes_per_loop_iteration / self._n_sim)
        CUTOFF = min(n_episodes_per_loop_iteration, 45)


        with self:
            if restore_path is not None:
                experiment.restore(restore_path)
            data_buffer = np.zeros(shape=(n_episodes_per_loop_iteration * lookback, self._episode_length), dtype=self._dtype)

            data = self.collect_episode_data_multi(
                n_data_collect * (lookback - 1),
                subkey,
                exploration_config(0),
            )

            data_buffer[n_episodes_per_loop_iteration:] = data
            for i in range(int(experiment_length_in_ep / n_episodes_per_loop_iteration)):
                n_episodes = i * n_episodes_per_loop_iteration

                key, subkey = random.split(subkey)

                exploration = exploration_config(n_episodes)
                data = self.collect_episode_data_multi(
                    n_data_collect,
                    subkey,
                    exploration,
                )
                training_return = np.mean(np.sum(data["rewards"], axis=1))

                start = (i % lookback) * n_episodes_per_loop_iteration
                stop = ((i % lookback) + 1) * n_episodes_per_loop_iteration
                data_buffer[start:stop] = data

                self.full_critic_training(data_buffer, n_critic_training_per_loop_iteration, subkey, i, tensorboard=tensorboard)
                self.full_actor_training(data_buffer, n_actor_training_per_loop_iteration, subkey, i, tensorboard=tensorboard)

                testing_data = self.collect_episode_data_multi(
                    max(20 // self._n_sim, 1),
                    subkey,
                    exploration_config.no_exploration,
                )
                testing_return = np.mean(np.sum(testing_data["rewards"], axis=1))

                if database is not None:
                    database.insert_result(
                        experiment_id,
                        loop_iteration=i,
                        episode_nb=i * n_episodes_per_loop_iteration,
                        training_episode_return=training_return,
                        testing_episode_return=testing_return,
                        exploration_param=exploration.param,
                    )

                if tensorboard is not None:
                    self.log_data(
                        tensorboard=tensorboard,
                        data=data[:CUTOFF],
                        iteration=i,
                        training_return=training_return,
                        testing_return=testing_return,
                        exploration=exploration,
                    )
