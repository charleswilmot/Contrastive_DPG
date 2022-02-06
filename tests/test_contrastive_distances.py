import jax
import jax.numpy as jnp
from typing import Callable


def contractive_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((jax.lax.stop_gradient(x1) - x2) ** 2, axis=-1)


def squared_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((x1 - x2) ** 2, axis=-1)


def euclidian_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(1e-6 + jnp.sum((x1 - x2) ** 2, axis=-1))


def distance_matrix(func: Callable, x: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(
        lambda x1: jax.vmap(
            lambda x2:
                func(x1, x2)
        )(x)
    )(x) # shape [D0, D0, D1, ..., DN-1]


def self_distances(x: jnp.ndarray) -> jnp.ndarray:
    # x shape [D0, D1, ..., DN]
    matrix = distance_matrix(squared_distance, x) # shape [D0, D0, D1, ..., DN-1]
    # matrix = distance_matrix(contractive_distance, x) # shape [D0, D0, D1, ..., DN-1]
    indices = jnp.triu_indices(x.shape[0], k=1)
    distances = matrix[indices] # shape [D0 * (D0 - 1) / 2, D1, ..., DN-1]
    return distances


def special_2(matrix, n, d_rep, d_stable, d_att, epsilon=1e-2):
    '''
    matrix: dimension [N, N, ...]
    '''
    squared_distances = jnp.sort(matrix, axis=0)[1:n+1] # [n, N, ...]
    repulsion = 1. / (squared_distances + epsilon)
    # return repulsion
    attraction = -repulsion
    # attraction = (distances - d_stable) ** 2 / 5
    rep = squared_distances < d_rep ** 2
    att  = jnp.logical_and(
        squared_distances < d_att ** 2,
        squared_distances > d_stable ** 2,
    )
    force = 0
    force = jnp.where(rep, repulsion, force)
    force = jnp.where(att, attraction, force)
    return force / 100



def special_4(matrix, n, d_min, d_max):
    matrix = jnp.sort(matrix, axis=0)[1:n+1] # [n, N, ...]
    return 2 * jax.nn.sigmoid(-matrix / d_min) + jax.nn.sigmoid(matrix - d_max)


def special_5(matrix, n, r_ilambda, r_height, a_ilambda, a_height, d_max):
    matrix = jnp.sort(matrix, axis=0)[1:n+1] # [n, N, ...]
    return (
        2 * r_height * jax.nn.sigmoid(-matrix / r_height / r_ilambda) +
        1 * a_height * jax.nn.sigmoid((matrix - d_max) / a_height / a_ilambda)
    )


def special_3(matrix, n, d_stable, d_att, epsilon=1e-2):
    '''
    matrix: dimension [N, N, ...]
    '''
    squared_distances = jnp.sort(matrix, axis=0)[1:n+1] # [n, N, ...]
    repulsion = 1. / (squared_distances + epsilon)
    # return repulsion
    attraction = -repulsion
    # attraction = (distances - d_stable) ** 2 / 5
    rep = squared_distances < d_rep ** 2
    att  = jnp.logical_and(
        squared_distances < d_att ** 2,
        squared_distances > d_stable ** 2,
    )
    force = 0
    force = jnp.where(rep, repulsion, force)
    force = jnp.where(att, attraction, force)
    return force / 100



def special(squared_distances, d_rep, d_stable, d_att, epsilon=1e-2):
    distances = jnp.sqrt(squared_distances)
    repulsion = 1. / (distances + epsilon)
    attraction = (distances - d_stable) ** 2 / 5
    rep = squared_distances < d_rep ** 2
    att  = jnp.logical_and(
        squared_distances < d_att ** 2,
        squared_distances > d_stable ** 2,
    )
    force = 0
    force = jnp.where(rep, repulsion, force)
    force = jnp.where(att, attraction, force)
    return force



def fun1():
    plt.ion()

    n_points = 100
    epsilon = 1e-2
    contrastive_coef = 0.1
    learning_rate = 1e-2
    weight_scaling = 1.5


    def loss(positions, t):
        scale = jnp.logspace(1, -1, n_points, base=weight_scaling).reshape((-1, 1))
        ###### simple parabola
        # distance_loss = jnp.sum(scale * positions ** 2) # shape [n_points * (n_points - 1) / 2]
        ###### 2-cups (fourth degree polynomial)
        # distance_loss = jnp.sum(scale *
        #     positions ** 2 * (positions - jnp.array([1, 0])) * (positions - jnp.array([-1, 0]))
        # ) # shape [n_points * (n_points - 1) / 2]
        ###### moving 2-cups (fourth degree polynomial)
        # distance_loss = jnp.sum(scale *
        #     positions ** 2 * (positions - jnp.array([jnp.cos(t), jnp.sin(t)])) * (positions - jnp.array([-jnp.cos(t), -jnp.sin(t)]))
        # ) # shape [n_points * (n_points - 1) / 2]
        # nothing
        distance_loss = 0

        # distance_loss = jnp.sum(scale * -jnp.cos(positions * 10) / 5) # shape [n_points * (n_points - 1) / 2]
        # contrastive_loss = jnp.sum(1. / (self_distances(positions) + epsilon)) / n_points
        contrastive_loss = jnp.sum(special_5(
            distance_matrix(euclidian_distance, positions),
            n=20,
            r_ilambda=0.03,
            r_height=2,
            a_ilambda=0.7,
            a_height=0.7,
            d_max=3,
        ))
        # contrastive_loss = jnp.sum(special_2(
        #     distance_matrix(squared_distance, positions),
        #     n,
        #     d_rep,
        #     d_stable,
        #     d_att,
        #     epsilon,
        # ))
        # contrastive_loss = jnp.sum(special(self_distances(positions), d_rep, d_stable, d_att, epsilon))
        # d_mean = (d_min + d_max) / 2
        # d_inter = (d_max - d_min) / 2
        # contrastive_loss = jnp.sum(
        #     jnp.clip(
        #         jnp.abs(d_mean - self_distances(positions)) - d_inter,
        #         a_min=0,
        #     )
        # )
        # contrastive_loss = 100 * jnp.sum(jnp.clip((d - self_distances(positions)) ** 2, a_max=d ** 2)) / n_points
        return distance_loss + contrastive_coef * contrastive_loss

    dloss_dpositions = jax.grad(loss)

    key = random.PRNGKey(0)
    positions = random.uniform(key, shape=(n_points, 2), minval=-1, maxval=1)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    path_collection = ax.scatter(positions[:, 0], positions[:, 1], c=np.linspace(0, 1, n_points))
    plt.draw()

    def display(positions):
        path_collection.set_offsets(positions)
        fig.canvas.draw_idle()
        plt.pause(1e-10)


    for i in range(5000):
        if i % 1 == 0:
            display(positions)
        _, key = random.split(key)
        positions -= learning_rate * dloss_dpositions(positions, i / 100)



def f(matrix, r_ilambda, r_height, a_ilambda, a_height, d_max):
    return (
        2 * r_height * jax.nn.sigmoid(-matrix / r_height / r_ilambda) +
        1 * a_height * jax.nn.sigmoid((matrix - d_max) / a_height / a_ilambda)
    )


def plot(ax, r_ilambda, r_height, a_ilambda, a_height, d_max, label=None):
    X = jnp.linspace(0, 5, 300)
    ax.plot(X, f(X, r_ilambda, r_height, a_ilambda, a_height, d_max), label=label)


if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for d_max in [2, 3, 4]:
        plot(ax,
            r_ilambda=0.06,
            r_height=2,
            a_ilambda=0.7,
            a_height=0.7,
            d_max=d_max,
            label=f'{d_max=}',
        )
    plt.show()

    fun1()
