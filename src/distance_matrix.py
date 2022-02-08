from typing import Callable
import jax.numpy as jnp
import jax


def contractive_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((jax.lax.stop_gradient(x1) - x2) ** 2, axis=-1)


def squared_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((x1 - x2) ** 2, axis=-1)


def euclidian_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(1e-6 + jnp.sum((x1 - x2) ** 2, axis=-1))


def distance_matrix_old(func: Callable, x: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(
        lambda x1: jax.vmap(
            lambda x2:
                func(x1, x2)
        )(x)
    )(x) # shape [D0, D0, D1, ..., DN-1]


def distance_matrix(func: Callable, x: jnp.ndarray) -> jnp.ndarray:
    return func(x[jnp.newaxis], x[:, jnp.newaxis]) # shape [D0, D0, D1, ..., DN-1]


def n_closest(func: Callable, x: jnp.ndarray, n: int) -> jnp.ndarray:
    return jnp.sort(func(x[jnp.newaxis], x[:, jnp.newaxis]), axis=1)[:, 1:n+1] # shape [D0, n, D1, ..., DN-1]


def self_distances(x: jnp.ndarray) -> jnp.ndarray:
    # x shape [D0, D1, ..., DN]
    matrix = distance_matrix(squared_distance, x) # shape [D0, D0, D1, ..., DN-1]
    indices = jnp.triu_indices(x.shape[0], k=1)
    distances = matrix[indices] # shape [D0 * (D0 - 1) / 2, D1, ..., DN-1]
    return distances


def potential_sink(matrix, n, r_ilambda, r_height, a_ilambda, a_height, d_max):
    matrix = jnp.sort(matrix, axis=0)[1:n+1] # [n, N, ...]
    return (
        2 * r_height * jax.nn.sigmoid(-matrix / r_height / r_ilambda) +
        1 * a_height * jax.nn.sigmoid((matrix - d_max) / a_height / a_ilambda)
    )
