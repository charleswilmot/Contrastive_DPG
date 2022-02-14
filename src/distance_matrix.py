from functools import partial
from typing import Callable
import jax.numpy as jnp
import numpy as np
import jax
import snow


@jax.jit
def contractive_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((jax.lax.stop_gradient(x1) - x2) ** 2, axis=-1)


@jax.jit
def squared_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((x1 - x2) ** 2, axis=-1)


@jax.jit
def euclidian_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(1e-6 + jnp.sum((x1 - x2) ** 2, axis=-1))


@jax.jit
def one_way_euclidian_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(1e-6 + jnp.sum((jax.lax.stop_gradient(x1) - x2) ** 2, axis=-1))


@partial(jax.jit, static_argnums=(0, 2))
def distance_matrix(func: Callable, x: jnp.ndarray, axis: int=0) -> jnp.ndarray:
    return func(
        jnp.expand_dims(x, axis=axis),
        jnp.expand_dims(x, axis=axis + 1),
    ) # shape [D0, D1, ..., Di, Di, ..., DN-2, DN-1]


@jax.jit
def softplus(x, smoothness):
    kx = x * smoothness
    return (kx + jnp.sqrt(kx ** 2 + 1)) / smoothness / 2


@jax.jit
def softplus_sink(x, d_min, d_max, slope_min, slope_max):
    CST = 10
    smoothness = CST / (d_max - d_min)
    return (
        softplus((d_min - x) * slope_min, smoothness / slope_min) +
        softplus((x - d_max) * slope_max, smoothness / slope_max)
    )


@partial(jax.jit, static_argnums=(0, 2))
def n_closest(func: Callable, x: jnp.ndarray, n: int) -> jnp.ndarray:
    return jnp.sort(func(x[jnp.newaxis], x[:, jnp.newaxis]), axis=1)[:, 1:n+1] # shape [D0, n, D1, ..., DN-1]


@partial(jax.jit, static_argnums=(1, 2 ,3, 4, 5, 6))
def potential_sink(x, n, r_ilambda, r_height, a_ilambda, a_height, d_max):
    matrix = jnp.sqrt(n_closest(squared_distance, x, n))
    matrix = jnp.sort(matrix, axis=1)[:, 1:n+1] # [N, n, ...]
    return (
        2 * r_height * jax.nn.sigmoid(-matrix / r_height / r_ilambda) +
        1 * a_height * jax.nn.sigmoid((matrix - d_max) / a_height / a_ilambda)
    )


def level_energy(subdata, level, level_config):
    dm = distance_matrix(one_way_euclidian_distance, subdata, axis=level) # shape [dim0, dim1, ..., dim(axis), dim(axis), ...A...]
    indices = tuple(slice(None) if i < level else slice(1, None) for i in range(level + 1))
    distances = dm[indices] # shape [dim0, dim1, ..., dim(axis) - 1, dim(axis), ...A...]
    energy = softplus_sink(distances, *level_config[1:])
    axis = tuple(range(level + 2))
    return jnp.mean(jnp.sum(energy, axis=axis))


@partial(jax.jit, static_argnums=(1,))
def get_hierarchization_loss(data, hierarchization_config):
    '''
    data: shape [N, ...A..., COORD_DIM]
    hierarchization_config: tuple((dim0, ...), (dim1, ...), ..., (dimn, ...))
    '''
    return jnp.sum(jnp.array(
        snow.map(data, hierarchization_config, level_energy)
    ))
