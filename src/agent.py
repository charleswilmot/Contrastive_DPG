from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import haiku as hk
from haiku import nets
import rlax
import optax
import logging
import sys
from distance_matrix import *
import matplotlib
matplotlib.use('Agg') # necessary to avoid conflict with Coppelia's Qt
import matplotlib.pyplot as plt
from plots import *
import numpy as np
from collections import defaultdict
import exploration


EPSILON = 1e-4


class Agent:
    def __init__(self,
            discount_factor,
            noise_magnitude_limit,
            hierarchization_config,
            hierarchization_coef,
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
        self._discount_factor = discount_factor # 0.96
        self._noise_magnitude_limit = noise_magnitude_limit # 0.2
        self._hierarchization_config = hierarchization_config
        self._hierarchization_coef = hierarchization_coef # 0.1
        self._actor_learning_rate = actor_learning_rate # 1e-3
        self._critic_learning_rate = critic_learning_rate # 1e-3
        self._actions_dim = actions_dim
        dims = jnp.array(tuple(dim for dim, _, _, _, _ in self._hierarchization_config))
        self._n_actors = int(np.prod(dims))
        self._logger.info(f'The number of actors to be simulated is {self._n_actors}')
        for level, (dim, d_min, d_max, slope_min, slope_max) in enumerate(self._hierarchization_config):
            self._logger.info(f'{level=} - {dim=} - {d_min=} - {d_max=} - {slope_min=} - {slope_max=}')
        self._logger.info("initializing... done")

        def actors(states):
            raw_net_output = hk.Sequential([
                nets.MLP([100, 100, self._actions_dim * self._n_actors]),
                jnp.tanh,
            ])(states)
            shape = (*(s for s in raw_net_output.shape[:-1]), self._n_actors, self._actions_dim)
            reshaped_net_output = jnp.reshape(
                raw_net_output,
                shape,
            ) # shape [..., N_ACTORS, ACTION_DIM]
            ndim = reshaped_net_output.ndim
            first_axis = ndim - 2
            axes = (first_axis, *(i for i in range(ndim) if i != first_axis))
            actions = jnp.transpose(reshaped_net_output, axes=axes)
            return actions

        self._policy_network = hk.without_apply_rng(hk.transform(actors))

        def critic(states, actions):
            if actions.ndim == states.ndim + 1:
                # action shape [N_ACTORS, ..., ACTION_DIM]
                # state shape [..., STATE_DIM]
                for state_s, action_s in zip(states.shape[:-1], actions.shape[1:-1]):
                    if state_s != action_s:
                        raise RuntimeError(f"states and actions are not broadcastable together... {states.shape=} {actions.shape=}")
                shape = (*(s for s in actions.shape[:-1]), states.shape[-1])
                states = jnp.broadcast_to(states[jnp.newaxis], shape)
            elif actions.ndim == states.ndim:
                pass
            else:
                raise RuntimeError(f"States and actions have incompatible dimensions: {states.ndim=} {actions.ndim=}")
            inp = jnp.concatenate([states, actions], axis=-1)
            return nets.MLP([100, 100, 1])(inp)[..., 0] # remove the last dim

        self._critic_network = hk.without_apply_rng(hk.transform(critic))

        self._actor_optimizer = optax.adam(self._actor_learning_rate)
        self._critic_optimizer = optax.adam(self._critic_learning_rate)
        self._logger.info("initializing... done")

    def init(self, dummy_states, dummy_actions, key):
        key, subkey = random.split(key)
        self._logger.info("constructing the parameters")
        self._actor_params_init = self._policy_network.init(subkey, dummy_states)
        key, subkey = random.split(key)
        self._critic_params_init = self._critic_network.init(subkey, dummy_states, dummy_actions)
        return (
            self._actor_params_init,
            self._actor_optimizer.init(self._actor_params_init),
            self._critic_params_init,
            self._critic_optimizer.init(self._critic_params_init),
        )

    @partial(jax.jit, static_argnums=(0, 6))
    def get_action(self, actor_params, critic_params, states, actions_tm2=None, actions_tm1=None, smoothing=0.0):
        ######### self._logger.debug(f'Inside get_action - {states.shape=}')
        actions = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, N_SIM, ACTION_DIM]
        ######### self._logger.debug(f'Inside get_action - {actions.shape=}')
        returns = self._critic_network.apply(critic_params, states, actions) # shape [N_ACTORS, N_SIM]
        if actions_tm1 is not None and actions_tm2 is not None:
            returns_actions_smoothing = jnp.sqrt(EPSILON + jnp.sum((
                +1 * jnp.expand_dims(actions_tm2, axis=0) +
                -2 * jnp.expand_dims(actions_tm1, axis=0) +
                +1 * actions
            ) ** 2, axis=-1))
            returns -= smoothing * returns_actions_smoothing
        ######### self._logger.debug(f'Inside get_action - {returns.shape=}')
        where = jnp.argmax(returns, axis=0)[jnp.newaxis, ..., jnp.newaxis] # shape [1, N_SIM, 1]
        policy_actions = jnp.take_along_axis(actions, where, axis=0)[0] # shape [N_SIM, ACTION_DIM]
        ######### self._logger.debug(f'Inside get_action - {policy_actions.shape=}')
        return policy_actions

    def get_explorative_action(self, actor_params, critic_params, states, key, expl, actions_tm2=None, actions_tm1=None, smoothing=0.0):
        ######### self._logger.debug(f'Inside get_action - {states.shape=}')
        actions = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, N_SIM, ACTION_DIM]
        ######### self._logger.debug(f'Inside get_action - {actions.shape=}')
        returns = self._critic_network.apply(critic_params, states, actions) # shape [N_ACTORS, N_SIM]
        if actions_tm1 is not None and actions_tm2 is not None:
            returns_actions_smoothing = jnp.sqrt(EPSILON + jnp.sum((
                +1 * jnp.expand_dims(actions_tm2, axis=0) +
                -2 * jnp.expand_dims(actions_tm1, axis=0) +
                +1 * actions
            ) ** 2, axis=-1))
            returns -= smoothing * returns_actions_smoothing
        ######### self._logger.debug(f'Inside get_action - {returns.shape=}')
        probabilities = exploration.get_actions_probabilities(
            returns,
            self._hierarchization_config,
            expl,
        )
        return exploration.sample(actions, probabilities, key)

    @partial(jax.jit, static_argnums=(0,))
    def actor_learning_step(self, learner_state, actor_params, critic_params, states):
        infos = {}
        actions_before = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        actor_loss_value, dloss_dtheta = jax.value_and_grad(self._actor_loss)(actor_params, critic_params, states)
        infos["mean_actor_loss"] = jnp.mean(actor_loss_value)
        updates, learner_state = self._actor_optimizer.update(dloss_dtheta, learner_state)
        actor_params = optax.apply_updates(actor_params, updates)
        actions_after = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
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
        return actor_params, learner_state, infos

    @partial(jax.jit, static_argnums=(0,))
    def actor_pretraining_step(self, learner_state, actor_params, states):
        infos = {}
        actions_before = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        hierarchization_loss_value, dloss_dtheta = jax.value_and_grad(self._hierarchization_loss)(actor_params, states)
        infos["mean_hierarchy_loss"] = jnp.mean(hierarchization_loss_value)
        updates, learner_state = self._actor_optimizer.update(dloss_dtheta, learner_state)
        actor_params = optax.apply_updates(actor_params, updates)
        return actor_params, learner_state, infos

    @partial(jax.jit, static_argnums=(0,))
    def critic_learning_step(self, learner_state, actor_params, critic_params, states, actions, rewards):
        infos = {}
        critic_loss_value, dloss_dtheta = jax.value_and_grad(self._critic_loss, argnums=1)(actor_params, critic_params, states, actions, rewards)
        infos["mean_critic_loss"] = jnp.mean(critic_loss_value)
        updates, learner_state = self._critic_optimizer.update(dloss_dtheta, learner_state)
        critic_params = optax.apply_updates(critic_params, updates)
        return critic_params, learner_state, infos

    @partial(jax.jit, static_argnums=(0,))
    def _hierarchization_loss(self, actor_params, states):
        actions = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        if self._hierarchization_coef != 0.0:
            return self._hierarchization_coef * get_hierarchization_loss(actions, self._hierarchization_config)
        else:
            return 0

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, actor_params, critic_params, states):
        actions = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        returns = self._critic_network.apply(critic_params, states, actions)
        ##### return ascent
        loss = -jnp.sum(jnp.mean(returns, axis=(-1, -2)))
        ##### contrastive loss
        if self._hierarchization_coef != 0.0:
            h_loss = get_hierarchization_loss(actions, self._hierarchization_config)
            loss += self._hierarchization_coef * h_loss
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, actor_params, critic_params, states, actions, rewards):
        '''
        states: dimension [BATCH, SEQUENCE, STATE_DIM]
        actions: dimension [BATCH, SEQUENCE, ACTION_DIM]
        rewards: dimension [BATCH, SEQUENCE]
        '''
        recomputed_actions = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        returns = self._critic_network.apply(critic_params, states, actions) # shape [BATCH, SEQUENCE]
        recomputed_returns = self._critic_network.apply(critic_params, states, recomputed_actions) # shape [N_ACTORS, BATCH, SEQUENCE]
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

    @partial(jax.jit, static_argnums=(0,))
    def get_log_batch(self, actor_params, critic_params, states, actions, rewards):
        log = {}
        recomputed_actions = self._policy_network.apply(actor_params, states) # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        sample_action_distance_matrix = distance_matrix(euclidian_distance, recomputed_actions[:, 0, 0]) # shape [N_ACTORS, N_ACTORS]
        returns = self._critic_network.apply(critic_params, states, actions) # shape [BATCH, SEQUENCE]
        recomputed_returns = self._critic_network.apply(critic_params, states, recomputed_actions) # shape [N_ACTORS, BATCH, SEQUENCE]
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
        td = target_returns - returns[..., :-1]

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

        sorted_distances = jnp.sort(sample_action_distance_matrix, axis=0)
        mean_closest_1 = jnp.mean(sorted_distances[1:2])
        mean_closest_5 = jnp.mean(sorted_distances[1:6])
        mean_closest_10 = jnp.mean(sorted_distances[1:11])
        mean_closest_20 = jnp.mean(sorted_distances[1:21])

        log["recomputed_actions"] = recomputed_actions                          # shape [N_ACTORS, BATCH, SEQUENCE, ACTION_DIM]
        log["returns"] = returns                                                # shape [BATCH, SEQUENCE]
        log["recomputed_returns"] = recomputed_returns                          # shape [N_ACTORS, BATCH, SEQUENCE]
        log["best_recomputed_returns"] = best_recomputed_returns                # shape [BATCH, SEQUENCE]
        log["policy_actions"] = policy_actions                                  # shape [BATCH, SEQUENCE, ACTION_DIM]
        log["reconstructed_noise"] = reconstructed_noise                        # shape [BATCH, SEQUENCE, ACTION_DIM]
        log["reconstructed_noise_magnitude"] = reconstructed_noise_magnitude    # shape [BATCH, SEQUENCE]
        log["lambdas"] = lambdas                                                # shape [BATCH, SEQUENCE]
        log["target_returns"] = target_returns
        log["td"] = td
        log["no_noise_target_returns"] = no_noise_target_returns
        log["no_noise_critic_loss"] = no_noise_critic_loss
        log["critic_loss"] = critic_loss
        log["mean_closest_1"] = mean_closest_1
        log["mean_closest_5"] = mean_closest_5
        log["mean_closest_10"] = mean_closest_10
        log["mean_closest_20"] = mean_closest_20
        return log

    # @partial(jax.jit, static_argnums=(0, 6))
    def log_data(self, actor_params, critic_params, training_data, testing_data, tensorboard, iteration, expl):
        '''
        states: dimension [BATCH, SEQUENCE, STATE_DIM]
        actions: dimension [BATCH, SEQUENCE, ACTION_DIM]
        rewards: dimension [BATCH, SEQUENCE]
        '''
        self._logger.info('logging to tensorboard')

        tensorboard.add_scalar('perf/training_return', jnp.mean(jnp.sum(training_data["rewards"], axis=-1)), iteration)
        tensorboard.add_scalar('perf/testing_return', jnp.mean(jnp.sum(testing_data["rewards"], axis=-1)), iteration)

        states = jnp.concatenate([training_data["states"], training_data["goals"]], axis=-1)
        actions = training_data["actions"]
        rewards = training_data["rewards"]

        BATCH_SIZE = 15
        N = len(states)
        slices = tuple(slice(i, i + BATCH_SIZE) for i in range(0, N, BATCH_SIZE))
        acc = defaultdict(int)

        for batch_num, s in enumerate(slices):
            self._logger.info(f'logging to tensorboard {batch_num+1}/{len(slices)}')
            st, ac, re = states[s], actions[s], rewards[s]
            log = self.get_log_batch(actor_params, critic_params, st, ac, re)
            size = len(st)

            acc["reconstructed_noise_magnitude"] += jnp.mean(jnp.sum(log["reconstructed_noise_magnitude"], axis=0))
            acc["reconstructed_noise_magnitude_max"] = max(acc["reconstructed_noise_magnitude_max"], jnp.max(log["reconstructed_noise_magnitude"]))
            acc["lambdas"] += jnp.mean(jnp.sum(log["lambdas"], axis=0))
            # todo: get rid of distance matrix, taking to much time for N_ACTORS > 200...
            # find a better metric to keep track of!
            acc["abs_td"] += jnp.mean(jnp.sum(jnp.abs(log["td"]), axis=0))
            acc["best_recomputed_returns_t0"] += jnp.sum(log["best_recomputed_returns"][:, 0])
            acc["best_recomputed_returns"] += jnp.mean(jnp.sum(log["best_recomputed_returns"], axis=0))
            acc["critic_loss"] += log["critic_loss"]
            acc["no_noise_critic_loss"] += log["no_noise_critic_loss"]
            acc["abs_actions"] += jnp.mean(jnp.sum(jnp.abs(log["recomputed_actions"]), axis=1), axis=(0, 1))
            acc["mean_closest_1"] += log["mean_closest_1"]
            acc["mean_closest_5"] += log["mean_closest_5"]
            acc["mean_closest_10"] += log["mean_closest_10"]
            acc["mean_closest_20"] += log["mean_closest_20"]




            # todo once only:
            if batch_num == 0:
                fig = plot_probabilities_hierarchy(
                    log["recomputed_returns"][:, 0, 0],
                    self._hierarchization_config,
                    expl,
                )
                tensorboard.add_figure(f'hierarchies/probs', fig, iteration, close=True)

                fig = plot_returns_hierarchy(
                    log["recomputed_returns"][:, 0, 0],
                    self._hierarchization_config,
                )
                tensorboard.add_figure(f'hierarchies/returns', fig, iteration, close=True)

                fig = plot_loss_hierarchy(
                    log["recomputed_actions"][:, 0, 0],
                    self._hierarchization_config,
                )
                tensorboard.add_figure(f'hierarchies/actions', fig, iteration, close=True)

                fig = plot_min_distance_hierarchy(
                    log["recomputed_actions"][:, 0, 0],
                    self._hierarchization_config,
                )
                tensorboard.add_figure(f'hierarchies/min_distances', fig, iteration, close=True)

                fig = plot_max_distance_hierarchy(
                    log["recomputed_actions"][:, 0, 0],
                    self._hierarchization_config,
                )
                tensorboard.add_figure(f'hierarchies/max_distances', fig, iteration, close=True)

                fig = plot_2D_projection(
                    log["recomputed_actions"][:, 0, 0],
                    self._hierarchization_config,
                )
                tensorboard.add_figure(f'tsne_projection', fig, iteration, close=True)

                tensorboard.add_histogram('noise/returns', log["recomputed_returns"][:, 0, 0], iteration)

                dims = tuple(dim for dim, _, _, _, _ in self._hierarchization_config)
                x = log["recomputed_actions"][..., 0, 0, :]
                if len(log["recomputed_actions"]) != int(np.prod(dims)):
                    raise RuntimeError(f"The number of points does not match with the description ({x.shape=} {self._hierarchization_config=})")
                shape = (*dims, *x.shape[1:]) # [dim0, dim1, ..., dimn, ...A..., COORD_DIM]
                action_hierarchy = jnp.reshape(x, shape) # [dim0, dim1, ..., dimn, COORD_DIM]
                n_levels = len(self._hierarchization_config)
                for l in range(n_levels):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    indices = tuple(slice(None) if j == l else 0 for j in range(n_levels))
                    subset = action_hierarchy[indices] # shape [dim(axis), COORD_DIM]
                    dm = distance_matrix(euclidian_distance, subset, axis=0) # shape [dim(axis), dim(axis)]
                    tri = jnp.triu_indices(dm.shape[0], k=1)
                    vmin = jnp.min(dm[tri])
                    cax = ax.matshow(dm, cmap='hot', vmin=vmin)
                    fig.colorbar(cax)
                    fig.tight_layout()
                    tensorboard.add_figure(f'sample_distance_matrix/level{l}', fig, iteration, close=True)



                IM_SIZE = 5
                for t in (0, states.shape[1] // 2):
                    return_data = log["recomputed_returns"][:, 0, t]
                    mini = jnp.min(return_data)
                    maxi = jnp.max(return_data)
                    metadata = jnp.digitize(return_data, jnp.linspace(mini, maxi, 5))
                    label_img = jnp.ones(shape=(log["recomputed_returns"].shape[0], 3, IM_SIZE, IM_SIZE), dtype=jnp.float32)
                    X = jnp.expand_dims(return_data, axis=-1) # [N_ACTORS, 1]
                    X = (X - mini) / (maxi - mini)
                    colors = (
                        jnp.array([[1, 0, 0]], dtype=jnp.float32) * X +
                        jnp.array([[0, 0, 1]], dtype=jnp.float32) * (1 - X)
                    ) # [N_ACTORS, 3]
                    label_img *= jnp.expand_dims(colors, axis=(2, 3))
                    tensorboard.add_embedding(
                        log["recomputed_actions"][:, 0, t],
                        metadata,
                        label_img=label_img,
                        tag=f'actions_t{t}',
                        global_step=iteration
                    )

                # todo once only:
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax.matshow(log["action_distance_matrix"][..., 0, 0])
                # ax.set_axis_off()
                # fig.tight_layout()
                # tensorboard.add_figure('sample_distance_matrix', fig, iteration, close=True)

                # accumulate the first batch of episodes

                fig = plt.figure()
                ax = fig.add_subplot(111)
                for i, rewrd in enumerate(re):
                    x = jnp.nonzero(rewrd)
                    y = rewrd[x] + i * 1.2
                    ax.scatter(x, y, color='k', alpha=0.5)
                ax.plot(jnp.array([0, 10]), jnp.stack([jnp.arange(BATCH_SIZE)] * 2) * 1.2, color='k')
                ax.plot(jnp.arange(BATCH_SIZE)[jnp.newaxis] * 1.2 + log["returns"][:BATCH_SIZE, ..., :-1].T, color='b')
                ax.plot(jnp.arange(BATCH_SIZE)[jnp.newaxis] * 1.2 + log["target_returns"][:BATCH_SIZE].T, color='r')
                ax.set_axis_off()
                fig.tight_layout()
                tensorboard.add_figure('returns_vs_targets', fig, iteration, close=True)

        tensorboard.add_scalar('noise/mean_magnitude', acc["reconstructed_noise_magnitude"] / N, iteration)
        tensorboard.add_scalar('noise/max_magnitude', acc["reconstructed_noise_magnitude_max"], iteration)
        tensorboard.add_scalar('noise/mean_lambda', acc["lambdas"] / N, iteration)

        tensorboard.add_scalar('noise/closest_1', acc["mean_closest_1"] / len(slices), iteration)
        tensorboard.add_scalar('noise/closest_5', acc["mean_closest_5"] / len(slices), iteration)
        tensorboard.add_scalar('noise/closest_10', acc["mean_closest_10"] / len(slices), iteration)
        tensorboard.add_scalar('noise/closest_20', acc["mean_closest_20"] / len(slices), iteration)

        tensorboard.add_scalar('perf/estimated_return_t0', acc["best_recomputed_returns_t0"] / N, iteration)
        tensorboard.add_scalar('perf/estimated_return_mean', acc["best_recomputed_returns"] / N, iteration)
        tensorboard.add_scalar('perf/critic_loss', acc["critic_loss"] / N, iteration)
        tensorboard.add_scalar('perf/no_noise_critic_loss', acc["no_noise_critic_loss"] / N, iteration)
        tensorboard.add_scalar('perf/mean_abs_td', acc["abs_td"] / N, iteration)

        tensorboard.add_scalar(f'{expl.type}/param', expl.param, iteration)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(
            jnp.arange(self._actions_dim),
            acc["abs_actions"] / N,
        )
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        tensorboard.add_figure('mean_abs_actions', fig, iteration, close=True)
