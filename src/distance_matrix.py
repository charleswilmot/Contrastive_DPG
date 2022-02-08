from functools import partial
from typing import Callable
import jax.numpy as jnp
import jax


@jax.jit
def contractive_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((jax.lax.stop_gradient(x1) - x2) ** 2, axis=-1)


@jax.jit
def squared_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((x1 - x2) ** 2, axis=-1)


@jax.jit
def euclidian_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(1e-6 + jnp.sum((x1 - x2) ** 2, axis=-1))


@partial(jax.jit, static_argnums=(0,))
def distance_matrix(func: Callable, x: jnp.ndarray) -> jnp.ndarray:
    return func(x[jnp.newaxis], x[:, jnp.newaxis]) # shape [D0, D0, D1, ..., DN-1]


@partial(jax.jit, static_argnums=(0, 2))
def n_closest(func: Callable, x: jnp.ndarray, n: int) -> jnp.ndarray:
    return jnp.sort(func(x[jnp.newaxis], x[:, jnp.newaxis]), axis=1)[:, 1:n+1] # shape [D0, n, D1, ..., DN-1]


# @partial(jax.jit, static_argnums=(1, 2 ,3, 4, 5, 6))
# def potential_sink_old(matrix, n, r_ilambda, r_height, a_ilambda, a_height, d_max):
#     matrix = jnp.sort(matrix, axis=0)[1:n+1] # [n, N, ...]
#     return (
#         2 * r_height * jax.nn.sigmoid(-matrix / r_height / r_ilambda) +
#         1 * a_height * jax.nn.sigmoid((matrix - d_max) / a_height / a_ilambda)
#     )


@partial(jax.jit, static_argnums=(1, 2 ,3, 4, 5, 6))
def potential_sink(x, n, r_ilambda, r_height, a_ilambda, a_height, d_max):
    matrix = jnp.sqrt(n_closest(squared_distance, x, n))
    matrix = jnp.sort(matrix, axis=1)[:, 1:n+1] # [N, n, ...]
    return (
        2 * r_height * jax.nn.sigmoid(-matrix / r_height / r_ilambda) +
        1 * a_height * jax.nn.sigmoid((matrix - d_max) / a_height / a_ilambda)
    )
