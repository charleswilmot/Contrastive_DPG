import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, '../src')
from distance_matrix import *
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.manifold import TSNE
from scipy.sparse.csgraph import minimum_spanning_tree
import jax.numpy as jnp


def plot_2D_projection(actions, in3D=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d' if in3D else None)

    actions_kd = TSNE(
        n_components=3 if in3D else 2,
        init='pca',
        learning_rate='auto'
    ).fit_transform(actions)

    if in3D:
        ax.scatter(actions_kd[:, 0], actions_kd[:, 1], actions_kd[:, 2])
    else:
        ax.scatter(actions_kd[:, 0], actions_kd[:, 1])

    dims = tuple(dim for dim, _, _, _, _ in hierarchization_config)
    shape = (*dims, *actions.shape[1:]) # [dim0, dim1, ..., dimn, ...A...]
    actions = np.reshape(actions, shape) # [dim0, dim1, ..., dimn, ...A...]
    shape = (*dims, *actions_kd.shape[1:]) # [dim0, dim1, ..., dimn, ...A...]
    actions_kd = np.reshape(actions_kd, shape)

    n_levels = len(hierarchization_config)
    for l in range(n_levels):
        indices = tuple(slice(None) if j <= l else 0 for j in range(n_levels))
        subset = actions[indices] # shape [dim0, ..., dim(l), ...A...]
        subset_kd = actions_kd[indices]
        dm = distance_matrix(euclidian_distance, subset, axis=l) # shape [dim0, dim1, ..., dim(level), dim(level)]
        flat_dm = dm.reshape((-1,) + dm.shape[l:])
        span_trees = np.stack([
            minimum_spanning_tree(small_dm).toarray()
            for small_dm in flat_dm
        ], axis=0).reshape(dm.shape)
        indices = span_trees.nonzero()
        indices_from = indices[:-1]
        indices_to = indices[:-2] + (indices[-1],)
        points_from = subset_kd[indices_from]
        points_to = subset_kd[indices_to]
        lines = np.stack([points_from, points_to], axis=-2)
        if in3D:
            lc = Line3DCollection(lines, colors=['b', 'r', 'g', 'k', 'o'][l], alpha=((n_levels - l) / n_levels) ** 2)
        else:
            lc = mc.LineCollection(lines, colors=['b', 'r', 'g', 'k', 'o'][l], alpha=((n_levels - l) / n_levels) ** 2)
        ax.add_collection(lc)

    return fig


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


if __name__ == '__main__':

    learning_rate = 2e-1

    k = 1.15
    SQRT2 = 1.41421356237
    SAFETY = 4
    minmax_factor = 1.5
    dmin2 = 0.6
    dmax2 = dmin2 * minmax_factor
    dmin1 = SAFETY * SQRT2 * (dmax2)
    dmax1 = dmin1 * minmax_factor
    dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
    dmax0 = dmin0 * minmax_factor

    hierarchization_config = (
        (5, dmin0, dmax0, 1 / k ** 0, 1 / k ** 0),
        (5, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (5, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )


    N_ACTORS = int(np.prod(tuple(dim for dim, _, _, _, _ in hierarchization_config)))

    recomputed_actions = np.random.uniform(size=(N_ACTORS, 7), low=-1, high=1)

    dloss_dactions = jax.value_and_grad(get_hierarchization_loss)

    for i in range(1000):
        loss_value, delta = dloss_dactions(recomputed_actions, hierarchization_config)
        print(i, loss_value)
        recomputed_actions -= learning_rate * delta


    plot_2D_projection(recomputed_actions)
    plt.show()
