import sys
sys.path.insert(1, '../src')
from distance_matrix import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random






def get_hierarchization_loss(x, desc):
    '''
    x: shape [N, ...A..., COORD_DIM]
    desc: tuple((dim0, ...), (dim1, ...), ..., (dimn, ...))
    '''
    dims = jnp.array(tuple(dim for dim, _, _, _, _ in desc))
    if len(x) != jnp.prod(dims):
        raise RuntimeError(f"The number of points does not match with the description ({x.shape=} {desc=})")
    shape = (*dims, *x.shape[1:]) # [dim0, dim1, ..., dimn, ...A..., COORD_DIM]
    x = jnp.reshape(x, shape) # [dim0, dim1, ..., dimn, ...A..., COORD_DIM]
    n_levels = len(desc)
    total_energy = 0
    for axis, (dim, d_min, d_max, slope_min, slope_max) in enumerate(desc):
        indices = tuple(slice(None) if i <= axis else 0 for i in range(n_levels))
        x_subset = x[indices] # shape [dim0, dim1, ..., dim(axis), ...A..., COORD_DIM]
        dm = distance_matrix(one_way_euclidian_distance, x[indices], axis=axis) # shape [dim0, dim1, ..., dim(axis), dim(axis), ...A...]
        indices = tuple(slice(None) if i < axis else slice(1, None) for i in range(axis + 1))
        distances = dm[indices] # shape [dim0, dim1, ..., dim(axis) - 1, dim(axis), ...A...]
        energy = softplus_sink(distances, d_min, d_max, slope_min, slope_max)
        axis = tuple(range(axis + 2))
        total_energy += jnp.mean(jnp.sum(energy, axis=axis))
    return total_energy


def fun1():


    k = 1
    l = 6
    dmin = 0.05
    dmax = dmin * l ** 0.3

    # points_hierarchy_desc = (
    #     (10, dmin * l ** 2, dmax * l ** 2, 1 / k ** 0, 1 / k ** 0),
    #     (10, dmin * l ** 1, dmax * l ** 1, 1 / k ** 1, 1 / k ** 1),
    #     (10, dmin * l ** 0, dmax * l ** 0, 1 / k ** 2, 1 / k ** 2),
    # )

    # SQRT2 = 2
    SQRT2 = 1.41421356237
    SAFETY = 2
    minmax_factor = 1.5
    dmin2 = 0.6
    dmax2 = dmin2 * minmax_factor
    dmin1 = SAFETY * SQRT2 * (dmax2)
    dmax1 = dmin1 * minmax_factor
    dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
    dmax0 = dmin0 * minmax_factor

    points_hierarchy_desc = (
        # (10, dmin0, dmax0, 1 / k ** 0, 1 / k ** 0),
        (60, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (2, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )

    # points_hierarchy_desc = (
    #     (10, dmin * l ** 1, dmax * l ** 1, 1 / k ** 1, 1 / k ** 0),
    #     (10, dmin * l ** 0, dmax * l ** 0, 1 / k ** 2, 1 / k ** 1),
    # )

    for d in points_hierarchy_desc: print(d)


    fig = plt.figure()
    for i, (dim, d_min, d_max, slope_min, slope_max) in enumerate(points_hierarchy_desc):
        ax = fig.add_subplot(1, len(points_hierarchy_desc), i + 1)
        X = jnp.linspace(0, d_max * 1.3, 200)
        Y = softplus_sink(X, d_min, d_max, slope_min, slope_max)
        ax.plot(X, Y)
    plt.show()

    plt.ion()

    dims = jnp.array(tuple(dim for dim, _, _, _, _ in points_hierarchy_desc))
    n_points = jnp.prod(dims)
    print(f"{n_points=}")
    hierarchization_coef = 1.70
    learning_rate = 1e-2
    weight_scaling = 1.5


    def loss(positions, t):
        scale = jnp.logspace(1, -1, n_points, base=weight_scaling).reshape((-1, 1))
        ###### simple parabola
        # distance_loss = jnp.sum(scale * positions ** 2) # shape [n_points * (n_points - 1) / 2]
        # distance_loss = jnp.sum(jnp.sqrt(jnp.sum(positions ** 2, axis=-1))) # shape [n_points * (n_points - 1) / 2]
        ###### 2-cups (fourth degree polynomial)
        # distance_loss = jnp.sum(scale *
        #     positions ** 2 * (positions - jnp.array([1, 0])) * (positions - jnp.array([-1, 0]))
        # ) # shape [n_points * (n_points - 1) / 2]
        ###### moving 2-cups (fourth degree polynomial)
        distance_loss = jnp.sum(scale *
            positions ** 2 * (positions - jnp.array([jnp.cos(t), jnp.sin(t)])) * (positions - jnp.array([-jnp.cos(t), -jnp.sin(t)]))
        ) # shape [n_points * (n_points - 1) / 2]
        # nothing
        # distance_loss = 0
        hierarchization_loss = get_hierarchization_loss(positions, points_hierarchy_desc)
        return distance_loss + hierarchization_coef * hierarchization_loss

    dloss_dpositions = jax.grad(loss)

    key = random.PRNGKey(0)
    positions = random.uniform(key, shape=(n_points, 2), minval=-1, maxval=1)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    path_collection = ax.scatter(positions[:, 0], positions[:, 1], c=jnp.linspace(0, 1, n_points))
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


def fun2():
    d_min, d_max, slope_min, slope_max = 1, 4, 1, 1
    X = jnp.linspace(0, 10, 200)

    fig = plt.figure()

    ax = fig.add_subplot(221)
    for d_min_ in [0.2, 0.6, 1, 3.5]:
        Y = softplus_sink(X, d_min_, d_max, slope_min, slope_max)
        ax.plot(X, Y)

    ax = fig.add_subplot(222)
    for d_max_ in [1.5, 3, 4, 5]:
        Y = softplus_sink(X, d_min, d_max_, slope_min, slope_max)
        ax.plot(X, Y)

    ax = fig.add_subplot(223)
    for slope_min_ in [1, 2, 3, 4]:
        Y = softplus_sink(X, d_min, d_max, slope_min_, slope_max)
        ax.plot(X, Y)

    ax = fig.add_subplot(224)
    for slope_max_ in [0.1, 0.5, 1, 2]:
        Y = softplus_sink(X, d_min, d_max, slope_min, slope_max_)
        ax.plot(X, Y)

    plt.show()

if __name__ == '__main__':
    fun1()
    # fun2()
