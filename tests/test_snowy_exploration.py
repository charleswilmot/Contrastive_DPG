import jax.numpy as jnp
import sys
sys.path.insert(1, '../src')
import snow


def get_actions_probabilities(returns, hierarchization_config, upsilon):
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
    return ret


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

    ps = get_actions_probabilities(recomputed_returns, hierarchization_config, upsilon=0.9)


    print(f'{ps=}\n\n')
