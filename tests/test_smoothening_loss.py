import jax
import jax.numpy as jnp
from typing import Callable
from scipy.special import binom


def get_derivatives(a, n):
    ret = [a]
    for i in range(n):
        ret.append(ret[-1][1:] - ret[-1][:-1])
        # ret.append((ret[-1][1:] - ret[-1][:-1]) / (i + 1))
    return ret[1:]




def get_derivative(a, n):
    # a has shape [N, ACTION_DIM]
    kernel = jnp.array([(-1) ** (n - i) * binom(n, i) for i in range(n + 1)]) # shape [n + 1]
    # kernel = jnp.stack([kernel] * a.shape[-1]) # shape [2, n + 1]
    return jax.lax.conv_general_dilated(
        jnp.expand_dims(a, axis=(0, -1)), # [BATCH, SEQUENCE, ACTION_DIM, C=1]
        jnp.expand_dims(kernel, axis=(-1, -2, -3)), # [SEQUENCE, 1, 1, 1]
        window_strides=[1, 1],
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), # N = 1, A = N, C = ACTION_DIM
    )[0, ..., 0] # remove batch and chanel dims
    # return jnp.convolve(a, kernel, mode=mode)


def fun1():
    plt.ion()

    n_points = 100
    smoothening_coef = 0.5
    learning_rate = 1e-2


    def loss(positions, t):
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
        epsilon = 1e-5
        # smoothening_loss = sum(jnp.sum(jnp.sqrt(epsilon + jnp.sum(x ** 2, axis=-1))) for x in get_derivatives(positions, n=2))
        derivative = get_derivative(positions, n=2)
        # print(derivative.shape)
        smoothening_loss = jnp.sum(jnp.sqrt(jnp.sum(derivative ** 2, axis=-1)))
        return distance_loss + smoothening_coef * smoothening_loss

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





if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt
    import numpy as np

    fun1()
