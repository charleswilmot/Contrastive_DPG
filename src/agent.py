from functools import partial
import jax
import jax.numpy as jnp
from jax import random
import haiku as hk
from haiku import nets
import rlax
import optax
import logging
import sys
from distance_matrix import *
import matplotlib
matplotlib.use('TkAgg') # necessary to avoid conflict with Coppelia's Qt
import matplotlib.pyplot as plt


EPSILON = 1e-4


class Agent:
    def __init__(self,
            n_actors,
            discount_factor,
            noise_magnitude_limit,
            contrastive_loss_coef,
            smoothing_loss_coef,
            actor_learning_rate,
            critic_learning_rate,
            actions_dim):
        self._logger = logging.getLogger("Agent")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info("initializing...")
        self._n_actors = n_actors
        self._discount_factor = discount_factor # 0.96
        self._noise_magnitude_limit = noise_magnitude_limit # 0.2
        self._contrastive_loss_coef = contrastive_loss_coef # 0.1
        self._smoothing_loss_coef = smoothing_loss_coef # 0.1
        self._actor_learning_rate = actor_learning_rate # 1e-3
        self._critic_learning_rate = critic_learning_rate # 1e-3
        self._actions_dim = actions_dim
        self._policy_network = hk.without_apply_rng(
            hk.transform(lambda states: hk.Sequential([
                nets.MLP([100, 100, self._actions_dim]),
                jnp.tanh
            ])(states))
        )
        self._critic_network = hk.without_apply_rng(
            hk.transform(lambda states, actions:
                nets.MLP([100, 100, 1])(jnp.concatenate([states, actions], axis=-1))[..., 0]
            )
        )
        self._actor_optimizer = optax.adam(self._actor_learning_rate)
        self._critic_optimizer = optax.adam(self._critic_learning_rate)
        self._logger.info("initializing... done")

    def init(self, dummy_states, dummy_actions, key):
        self._logger.info("constructing the parameters")
        self._actors_params_init = [
            self._policy_network.init(subkey, dummy_states)
            for subkey in random.split(key, self._n_actors)
        ]
        key, subkey = random.split(key)
        self._critic_params_init = self._critic_network.init(subkey, dummy_states, dummy_actions)
        return (
            self._actors_params_init,
            self._actor_optimizer.init(self._actors_params_init),
            self._critic_params_init,
            self._critic_optimizer.init(self._critic_params_init),
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, actors_params, critic_params, states, actions_tm2=None, actions_tm1=None):
        ######### self._logger.debug(f'Inside get_action - {states.shape=}')
        actions = jnp.stack([
            self._policy_network.apply(actor_params, states) # shape [N_SIM, ACTION_DIM]
            for actor_params in actors_params
        ]) # shape [N_ACTORS, N_SIM, ACTION_DIM]
        ######### self._logger.debug(f'Inside get_action - {actions.shape=}')
        returns = jnp.stack([
            self._critic_network.apply(critic_params, states, act)
            for act in actions
        ]) # shape [N_ACTORS, N_SIM]
        if actions_tm1 is not None and actions_tm2 is not None:
            returns_actions_smoothing = jnp.sqrt(EPSILON + jnp.sum((
                +1 * jnp.expand_dims(actions_tm2, axis=0) +
                -2 * jnp.expand_dims(actions_tm1, axis=0) +
                +1 * actions
            ) ** 2, axis=-1))
        else:
            returns_actions_smoothing = jnp.zeros_like(returns)
        returns -= self._smoothing_loss_coef * returns_actions_smoothing
        ######### self._logger.debug(f'Inside get_action - {returns.shape=}')
        where = jnp.argmax(returns, axis=0)[jnp.newaxis, ..., jnp.newaxis] # shape [1, N_SIM, 1]
        policy_actions = jnp.take_along_axis(actions, where, axis=0)[0] # shape [N_SIM, ACTION_DIM]
        ######### self._logger.debug(f'Inside get_action - {policy_actions.shape=}')
        return policy_actions

    def get_explorative_action(self, actors_params, critic_params, states, key):
        index = random.randint(key, shape=(), minval=0, maxval=self._n_actors)
        return self._policy_network.apply(actors_params[index], states)

    @partial(jax.jit, static_argnums=(0,))
    def actor_learning_step(self, learner_state, actors_params, critic_params, states):
        infos = {}
        actions_before = jnp.stack([
            self._policy_network.apply(actor_params, states) # shape [BATCH, SEQUENCE, ACTION_DIM]
            for actor_params in actors_params
        ]) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        actor_loss_value, dloss_dtheta = jax.value_and_grad(self._actor_loss)(actors_params, critic_params, states)
        infos["mean_actor_loss"] = jnp.mean(actor_loss_value)
        updates, learner_state = self._actor_optimizer.update(dloss_dtheta, learner_state)
        actors_params = optax.apply_updates(actors_params, updates)
        actions_after = jnp.stack([
            self._policy_network.apply(actor_params, states) # shape [BATCH, SEQUENCE, ACTION_DIM]
            for actor_params in actors_params
        ]) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        delta_actions_norm = jnp.sqrt(EPSILON + (actions_after - actions_before) ** 2)
        infos["max(|delta_actions|)"] = jnp.max(jnp.sum(delta_actions_norm, axis=-1))
        infos["mean(|delta_actions|)"] = jnp.mean(jnp.sum(delta_actions_norm, axis=-1))
        d2actions_dt2_before = jnp.mean(jnp.sqrt(EPSILON + jnp.sum((
            +1 * actions_before[:, :, 0:-2] +
            -2 * actions_before[:, :, 1:-1] +
            +1 * actions_before[:, :, 2:]
        ) ** 2, axis=-1)))
        d2actions_dt2_after = jnp.mean(jnp.sqrt(EPSILON + jnp.sum((
            +1 * actions_after[:, :, 0:-2] +
            -2 * actions_after[:, :, 1:-1] +
            +1 * actions_after[:, :, 2:]
        ) ** 2, axis=-1)))
        infos["delta_mean(|d2actions_dt2|)"] = d2actions_dt2_after - d2actions_dt2_before
        infos["mean(|d2actions_dt2|)"] = d2actions_dt2_after
        return actors_params, learner_state, infos

    @partial(jax.jit, static_argnums=(0,))
    def critic_learning_step(self, learner_state, actors_params, critic_params, states, actions, rewards):
        infos = {}
        critic_loss_value, dloss_dtheta = jax.value_and_grad(self._critic_loss, argnums=1)(actors_params, critic_params, states, actions, rewards)
        infos["mean_critic_loss"] = jnp.mean(critic_loss_value)
        updates, learner_state = self._critic_optimizer.update(dloss_dtheta, learner_state)
        critic_params = optax.apply_updates(critic_params, updates)
        return critic_params, learner_state, infos

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, actors_params, critic_params, states):
        actions = jnp.stack([
            self._policy_network.apply(actor_params, states) # shape [BATCH, SEQUENCE, ACTION_DIM]
            for actor_params in actors_params
        ]) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        returns = jnp.stack([
            self._critic_network.apply(critic_params, states, act)
            for act in actions
        ]) # shape [N_ACTORS, BATCH, SEQUENCE]

        ##### return ascent
        loss = -jnp.sum(jnp.mean(returns, axis=(-1, -2)))

        ##### actions smoothing
        returns_actions_smoothing = jnp.sqrt(EPSILON + jnp.sum((
            +1 * actions[:, :, 0:-2] +
            -2 * actions[:, :, 1:-1] +
            +1 * actions[:, :, 2:]
        ) ** 2, axis=-1)) # [N_ACTORS, BATCH, SEQUENCE - 2]
        loss += self._smoothing_loss_coef * jnp.sum(jnp.mean(returns_actions_smoothing, axis=(-1, -2)))

        ##### contrastive loss
        matrix = distance_matrix(euclidian_distance, actions)
        loss += self._contrastive_loss_coef * jnp.sum(potential_sink(matrix,
            n=5,
            r_ilambda=0.06,
            r_height=2,
            a_ilambda=0.7,
            a_height=0.7,
            d_max=3,
        ))
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, actors_params, critic_params, states, actions, rewards):
        '''
        states: dimension [BATCH, SEQUENCE, STATE_DIM]
        actions: dimension [BATCH, SEQUENCE, ACTION_DIM]
        rewards: dimension [BATCH, SEQUENCE]
        '''
        recomputed_actions = jnp.stack([
            self._policy_network.apply(actor_params, states) # shape [BATCH, SEQUENCE, ACTION_DIM]
            for actor_params in actors_params
        ]) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        returns = self._critic_network.apply(critic_params, states, actions) # shape [BATCH, SEQUENCE]
        recomputed_returns = jnp.stack([
            self._critic_network.apply(critic_params, states, recomputed_act)
            for recomputed_act in recomputed_actions
        ]) # shape [N_ACTORS, BATCH, SEQUENCE]
        where = jnp.argmax(recomputed_returns, axis=0)[jnp.newaxis, ..., jnp.newaxis] # shape [BATCH, SEQUENCE]
        best_recomputed_returns = jnp.take_along_axis(recomputed_returns, where[..., 0], axis=0)[0] # shape [BATCH, SEQUENCE]
        policy_actions = jnp.take_along_axis(recomputed_actions, where, axis=0)[0] # shape [BATCH, SEQUENCE, ACTION_DIM]
        ########################################################################

        # first: determine which actions are exploratory (ie. define the lambdas)
        reconstructed_noise = actions - policy_actions
        reconstructed_noise_magnitude = jnp.sqrt(EPSILON + jnp.sum(reconstructed_noise ** 2, axis=-1)) # shape [BATCH, SEQUENCE]
        LN_2 = 0.693147
        lambdas = jnp.exp(-LN_2 * reconstructed_noise_magnitude / self._noise_magnitude_limit)

        # second: compute the target returns
        discounts = jnp.full_like(rewards, self._discount_factor)
        batched = jax.vmap(
            lambda rewards, discounts, best_recomputed_returns, lambdas:
                rlax.lambda_returns(
                    rewards[..., :-1],
                    discounts[..., :-1],
                    best_recomputed_returns[..., 1:],
                    lambdas[..., :-1],
                    stop_target_gradients=True
                )
        )
        target_returns = batched(rewards, discounts, best_recomputed_returns, lambdas)

        loss = jnp.mean(rlax.l2_loss(returns[..., :-1], target_returns))
        # third: return the l2 loss / hubber loss
        return loss # shape [BATCH, SEQUENCE]

    # @partial(jax.jit, static_argnums=(0, 6))
    def log_data(self, actors_params, critic_params, states, actions, rewards, tensorboard, iteration):
        '''
        states: dimension [BATCH, SEQUENCE, STATE_DIM]
        actions: dimension [BATCH, SEQUENCE, ACTION_DIM]
        rewards: dimension [BATCH, SEQUENCE]
        '''
        recomputed_actions = jnp.stack([
            self._policy_network.apply(actor_params, states) # shape [BATCH, SEQUENCE, ACTION_DIM]
            for actor_params in actors_params
        ]) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        returns = self._critic_network.apply(critic_params, states, actions) # shape [BATCH, SEQUENCE]
        recomputed_returns = jnp.stack([
            self._critic_network.apply(critic_params, states, recomputed_act)
            for recomputed_act in recomputed_actions
        ]) # shape [N_ACTORS, BATCH, SEQUENCE]
        where = jnp.argmax(recomputed_returns, axis=0)[jnp.newaxis, ..., jnp.newaxis] # shape [BATCH, SEQUENCE]
        best_recomputed_returns = jnp.take_along_axis(recomputed_returns, where[..., 0], axis=0)[0] # shape [BATCH, SEQUENCE]
        policy_actions = jnp.take_along_axis(recomputed_actions, where, axis=0)[0] # shape [BATCH, SEQUENCE, ACTION_DIM]
        ########################################################################

        # first: determine which actions are exploratory (ie. define the lambdas)
        reconstructed_noise = actions - policy_actions
        reconstructed_noise_magnitude = jnp.sqrt(EPSILON + jnp.sum(reconstructed_noise ** 2, axis=-1)) # shape [BATCH, SEQUENCE]
        LN_2 = 0.693147
        lambdas = jnp.exp(-LN_2 * reconstructed_noise_magnitude / self._noise_magnitude_limit)


        # second: compute the target returns
        discounts = jnp.full_like(rewards, self._discount_factor)
        batched = jax.vmap(
            lambda rewards, discounts, best_recomputed_returns, lambdas:
                rlax.lambda_returns(
                    rewards[..., :-1],
                    discounts[..., :-1],
                    best_recomputed_returns[..., 1:],
                    lambdas[..., :-1],
                    stop_target_gradients=True
                )
        )
        target_returns = batched(rewards, discounts, best_recomputed_returns, lambdas)

        # third: compute the l2 loss / hubber loss
        critic_loss = jnp.mean(rlax.l2_loss(returns[..., :-1], target_returns))

        batched = jax.vmap(
            lambda rewards, discounts, best_recomputed_returns:
                rlax.lambda_returns(
                    rewards[..., :-1],
                    discounts[..., :-1],
                    best_recomputed_returns[..., 1:],
                    1.0,
                    stop_target_gradients=True
                )
        )
        no_noise_target_returns = batched(rewards, discounts, returns)
        no_noise_critic_loss = jnp.mean(rlax.l2_loss(returns[..., :-1], no_noise_target_returns))


        self._logger.info('logging to tensorboard')

        tensorboard.add_scalar('noise/mean_magnitude', jnp.mean(reconstructed_noise_magnitude), iteration)
        tensorboard.add_scalar('noise/max_magnitude', jnp.max(reconstructed_noise_magnitude), iteration)
        tensorboard.add_scalar('noise/mean_lambda', jnp.mean(lambdas), iteration)
        tensorboard.add_scalar('noise/std_lambda', jnp.std(lambdas), iteration)
        tensorboard.add_histogram('noise/magnitudes', reconstructed_noise_magnitude, iteration)
        tensorboard.add_histogram('noise/lambdas', lambdas, iteration)
        action_distance_matrix = distance_matrix(euclidian_distance, recomputed_actions) # shape [N_ACTORS, N_ACTORS, BATCH, SEQUENCE]
        sorted_distances = jnp.sort(action_distance_matrix, axis=0)
        closest_5 = sorted_distances[1:6]
        closest_10 = sorted_distances[1:11]
        tensorboard.add_scalar('noise/closest_5_mean_dist', jnp.mean(closest_5), iteration)
        tensorboard.add_scalar('noise/closest_10_mean_dist', jnp.mean(closest_10), iteration)
        IM_SIZE = 5
        for t in (0, recomputed_actions.shape[2] // 2):
            return_data = recomputed_returns[:, 0, t]
            mini = jnp.min(return_data)
            maxi = jnp.max(return_data)
            metadata = jnp.digitize(return_data, jnp.linspace(mini, maxi, 5))
            label_img = jnp.ones(shape=(recomputed_returns.shape[0], 3, IM_SIZE, IM_SIZE), dtype=jnp.float32)
            X = jnp.expand_dims(return_data, axis=-1) # [N_ACTORS, 1]
            X = (X - mini) / (maxi - mini)
            colors = (
                jnp.array([[1, 0, 0]], dtype=jnp.float32) * X +
                jnp.array([[0, 0, 1]], dtype=jnp.float32) * (1 - X)
            ) # [N_ACTORS, 3]
            label_img *= jnp.expand_dims(colors, axis=(2, 3))
            tensorboard.add_embedding(
                recomputed_actions[:, 0, t],
                metadata,
                label_img=label_img,
                tag=f'actions_t{t}',
                global_step=iteration
            )

        actions_smoothing = jnp.sqrt(EPSILON + jnp.sum((
            +1 * recomputed_actions[:, :, 0:-2] +
            -2 * recomputed_actions[:, :, 1:-1] +
            +1 * recomputed_actions[:, :, 2:]
        ) ** 2, axis=-1)) # [N_ACTORS, BATCH, SEQUENCE - 2]
        tensorboard.add_scalar('actions/mean_smoothing', jnp.mean(actions_smoothing), iteration)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(action_distance_matrix[..., 0, 0])
        ax.set_axis_off()
        fig.tight_layout()
        tensorboard.add_figure('sample_distance_matrix', fig, iteration, close=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(
            jnp.arange(recomputed_actions.shape[-1]),
            jnp.mean(jnp.abs(recomputed_actions), axis=tuple(range(recomputed_actions.ndim - 1)))
        )
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        tensorboard.add_figure('mean_abs_actions', fig, iteration, close=True)

        # tensorboard.add_scalar('perf/success_rate', jnp.mean(
        #     jnp.abs((data["states"][..., -1, -self._registers_dim:] / 0.5 + 0.5 - data["goals"][..., -1, :]) < 1e-3).all()
        # ) * 100, iteration)
        tensorboard.add_scalar('perf/episode_return', jnp.mean(jnp.sum(rewards, axis=1)), iteration)
        tensorboard.add_scalar('perf/estimated_return_t0', jnp.mean(best_recomputed_returns[:, 0]), iteration)
        tensorboard.add_scalar('perf/estimated_return_mean', jnp.mean(best_recomputed_returns), iteration)
        tensorboard.add_scalar('perf/critic_loss', critic_loss, iteration)
        tensorboard.add_scalar('perf/no_noise_critic_loss', no_noise_critic_loss, iteration)
        tensorboard.add_scalar('perf/mean_abs_td', jnp.mean(jnp.abs(target_returns - returns[..., :-1])), iteration)
        tensorboard.add_histogram('perf/td', target_returns - returns[..., :-1], iteration)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        N_LINES = 15
        for i, rewrd in enumerate(rewards[:N_LINES]):
            x = jnp.nonzero(rewrd)
            y = rewrd[x] + i * 1.2
            ax.scatter(x, y, color='k', alpha=0.5)
        ax.plot(jnp.array([0, 10]), jnp.stack([jnp.arange(N_LINES)] * 2) * 1.2)
        ax.plot(jnp.arange(N_LINES)[jnp.newaxis] * 1.2 + returns[:N_LINES, ..., :-1].T, color='b')
        ax.plot(jnp.arange(N_LINES)[jnp.newaxis] * 1.2 + target_returns[:N_LINES].T, color='r')
        ax.set_axis_off()
        fig.tight_layout()
        tensorboard.add_figure('returns_vs_targets', fig, iteration, close=True)

        dreturn_daction = jax.vmap(
            jax.vmap(
                jax.grad(self._critic_network.apply, argnums=2),
                in_axes=(None, 0, 0),
            ),
            in_axes=(None, 0, 0),
        )
        grads = jnp.stack([
            dreturn_daction(critic_params, states, recomputed_act)
            for recomputed_act in recomputed_actions
        ]) # shape [N_ACTOR, BATCH, SEQUENCE, ACTION_DIM]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(
            jnp.reshape(grads, (-1, self._actions_dim)).T
        )
        ax.set_ylim(-0.4, 0.4)
        fig.tight_layout()
        tensorboard.add_figure('gradient_distribution', fig, iteration, close=True)
