import jax
import jax.numpy as jnp
import snow
from collections import namedtuple
import numpy as np


Exploration = namedtuple("Exploration", ['type', 'param'])


class ExplorationConfig:
    def __init__(self, type, N, interpolation_type, upsilon_t0, upsilon_tN,
        exploration_prob_t0, exploration_prob_tN, softmax_temperature_t0,
        softmax_temperature_tN):

        self._type = type
        self._N = N
        self._interpolation_type = interpolation_type
        self._upsilon_t0 = upsilon_t0
        self._upsilon_tN = upsilon_tN
        self._exploration_prob_t0 = exploration_prob_t0
        self._exploration_prob_tN = exploration_prob_tN
        self._softmax_temperature_t0 = softmax_temperature_t0
        self._softmax_temperature_tN = softmax_temperature_tN

        if self._type == 'upsilon':
            self._param_t0 = self._upsilon_t0
            self._param_tN = self._upsilon_tN
        elif self._type == 'exploration_prob':
            self._param_t0 = self._exploration_prob_t0
            self._param_tN = self._exploration_prob_tN
        elif self._type == 'softmax_temperature':
            self._param_t0 = self._softmax_temperature_t0
            self._param_tN = self._softmax_temperature_tN

        if self._interpolation_type == 'linear':
            self._func = lambda x: np.clip(x / self._N, 0, 1)
        elif self._interpolation_type == 'cosine':
            self._func = lambda x: (1 - np.cos(np.pi * np.clip(x / self._N, 0, 1))) / 2

    def __call__(self, episode_nb):
        param = self.interpolation(episode_nb)
        return Exploration(self._type, param)

    def interpolation(self, episode_nb):
        return self._func(episode_nb) * (self._param_tN - self._param_t0) + self._param_t0

    @property
    def no_exploration(self):
        return Exploration('exploration_prob', 0.0)


def get_actions_probabilities(returns, hierarchization_config, expl):
    if expl.type == 'upsilon':
        return get_upsilon_actions_probabilities(
            returns,
            hierarchization_config,
            expl.param,
        )
    elif expl.type == 'exploration_prob':
        return get_exploration_probabilities(
            returns,
            expl.param,
        )
    elif expl.type == 'softmax_temperature':
        return get_softmax_probabilities(
            returns,
            expl.param,
        )


def get_exploration_probabilities(returns, prob):
    n = returns.shape[0]
    return jnp.where(returns == jnp.max(returns, axis=0)[jnp.newaxis], 1 - prob, prob / n)


def get_softmax_probabilities(returns, temperature):
    return jax.nn.softmax(returns / temperature, axis=0, initial=0.0)


def get_upsilon_actions_probabilities(returns, hierarchization_config, upsilon):
    reshaped_returns = snow.reshape(returns, hierarchization_config)
    n_levels = len(hierarchization_config)
    maxs = []
    for level, (size, _, _, _, _) in enumerate(hierarchization_config):
        axis = tuple(i for i in range(n_levels) if i >= level)
        maxs.append(jnp.max(reshaped_returns, axis=axis, keepdims=True))
    maxs.append(reshaped_returns)
    ret = 1
    for (size, _, _, _, _), m0, m1 in zip(hierarchization_config, maxs[:-1], maxs[1:]):
        cond = m0 == m1
        low_p = (1 - upsilon) / size
        high_p = upsilon * (1 - 1 / size) + 1 / size
        ret *= jnp.where(cond, high_p, low_p)
    return ret.reshape(returns.shape)


def sample(actions, probabilities, key):
    # actions shape [N_ACTORS, BATCH, ACTION_DIM]
    if actions.ndim != 3:
        raise RuntimeError(f"Cannot sample from actions, ndim={actions.ndim} expected 3")
    # probabilities shape [N_ACTORS, BATCH]
    if probabilities.ndim != 2:
        raise RuntimeError(f"Cannot sample from probabilities, ndim={probabilities.ndim} expected 2")
    p_cuml = jnp.cumsum(probabilities, axis=0) # shape [N_ACTORS, BATCH]
    shape = probabilities.shape[1:]
    r = p_cuml[-1] * (1 - jax.random.uniform(key, shape)) # shape [BATCH]
    cond = p_cuml > r[jnp.newaxis]
    indices = jnp.argmax(cond, axis=0)[jnp.newaxis, ..., jnp.newaxis] # shape [1, BATCH, 1]
    return jnp.take_along_axis(actions, indices, axis=0)[0] # shape [BATCH, ACTION_DIM]


if __name__ == '__main__':
    import numpy as np

    k = 1.15
    l = 2
    dmin = 0.8
    dmax = dmin * l ** 0.6
    hierarchization_config = (
        (5, dmin * l ** 2, dmax * l ** 2, 1 / k ** 0, 1 / k ** 0),
        (5, dmin * l ** 1, dmax * l ** 1, 1 / k ** 1, 1 / k ** 1),
        (5, dmin * l ** 0, dmax * l ** 0, 1 / k ** 2, 1 / k ** 2),
    )
    N_ACTORS = int(np.prod(tuple(dim for dim, _, _, _, _ in hierarchization_config)))

    # recomputed_returns = np.random.uniform(size=(N_ACTORS,), high=12)
    recomputed_returns = np.linspace(1, 0, N_ACTORS)
    recomputed_returns = np.stack([np.linspace(1, 0, N_ACTORS)] * 2, axis=1)
    recomputed_actions = np.stack([np.random.uniform(size=(N_ACTORS, 7), low=-1, high=1)], axis=1)

    ps = get_actions_probabilities(recomputed_returns, hierarchization_config, upsilon=0.0)
    # print(f'{ps=}\n\n')

    key = jax.random.PRNGKey(3)
    actions = sample(recomputed_actions, ps, key)
