from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from distance_matrix import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.manifold import TSNE
from scipy.sparse.csgraph import minimum_spanning_tree
import snow
import exploration


def plot_hierarchy(data, hierarchization_config, func=lambda subdata, level, level_config: subdata, cmap='hot'):
    SIZE = 0.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    by_level = snow.map(data, hierarchization_config, func)
    cmap = plt.get_cmap(cmap)
    for level, processed in enumerate(by_level):
        vmin = np.min(processed)
        vmax = np.max(processed)
        norm = Normalize(vmin=vmin, vmax=vmax)
        n = np.prod(tuple(dim for dim, _, _, _, _ in hierarchization_config[:level+1]))
        vals = np.ones(n)
        colors = cmap(norm(processed).flatten())
        p = ax.pie(
            vals,
            radius=SIZE * (level + 1),
            colors=colors,
            wedgeprops=dict(
                width=SIZE - 0.02,
                edgecolor='w'
            )
        )
        cax = divider.append_axes("right", size="5%", pad=0.6)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, cax=cax)
        cb.set_label(f'{level=}')
    return fig


def plot_2D_projection(actions, hierarchization_config, in3D=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d' if in3D else None)
    n_levels = len(hierarchization_config)

    actions_kd = TSNE(
        n_components=3 if in3D else 2,
        init='pca',
        learning_rate='auto'
    ).fit_transform(actions)

    if in3D:
        ax.scatter(actions_kd[:, 0], actions_kd[:, 1], actions_kd[:, 2])
    else:
        ax.scatter(actions_kd[:, 0], actions_kd[:, 1])

    def get_minimum_spanning_trees_from_to_indices(subdata, level, level_config):
        dm = distance_matrix(euclidian_distance, subdata, axis=level) # shape [dim0, dim1, ..., dim(level), dim(level)]
        flat_dm = dm.reshape((-1,) + dm.shape[level:])
        span_trees = np.stack([
            minimum_spanning_tree(small_dm).toarray()
            for small_dm in flat_dm
        ], axis=0).reshape(dm.shape)
        indices = span_trees.nonzero()
        indices_from = indices[:-1]
        indices_to = indices[:-2] + (indices[-1],)
        return indices_from, indices_to

    from_to = snow.map(actions, hierarchization_config, get_minimum_spanning_trees_from_to_indices)
    actions_kd_snowflake = snow.map(actions_kd, hierarchization_config)

    for level, (act_kd, (i_from, i_to)) in enumerate(zip(actions_kd_snowflake, from_to)):
        points_from = act_kd[i_from]
        points_to = act_kd[i_to]
        lines = np.stack([points_from, points_to], axis=-2)
        LineCollection = Line3DCollection if in3D else mc.LineCollection
        lc = LineCollection(
            lines,
            colors=['b', 'r', 'g', 'k', 'o'][level],
            alpha=((n_levels - level) / n_levels), # ** 2
        )
        ax.add_collection(lc)

    return fig


def plot_probabilities_hierarchy(recomputed_returns, hierarchization_config, expl, cmap='hot'):
    probs = exploration.get_actions_probabilities(recomputed_returns[:, jnp.newaxis], hierarchization_config, expl)[:, 0]
    return plot_hierarchy(
        probs,
        hierarchization_config,
        # func=lambda subdata, _a, _b: np.sort(subdata, axis=-1),
    )


def plot_returns_hierarchy(recomputed_returns, hierarchization_config, cmap='hot'):
    return plot_hierarchy(
        recomputed_returns,
        hierarchization_config,
        # func=lambda subdata, _a, _b: np.sort(subdata, axis=-1),
    )


def plot_loss_hierarchy(recomputed_returns, hierarchization_config, cmap='hot'):
    def func(subdata, level, level_config):
        # subdata shape: [d0, ..., d(level), ACTION_DIM]
        dm = distance_matrix(euclidian_distance, subdata, axis=level) # shape [dim0, dim1, ..., dim(level), dim(level)]
        energy = softplus_sink(dm, *hierarchization_config[level][1:])
        N = level_config[0]
        # return np.sort(np.sum(energy, axis=-1), axis=-1) / (N - 1)
        return np.sum(energy, axis=-1) / (N - 1)

    return plot_hierarchy(
        recomputed_returns,
        hierarchization_config,
        func=func,
    )


def plot_min_distance_hierarchy(recomputed_returns, hierarchization_config, cmap='hot'):
    def func(subdata, level, level_config):
        # subdata shape: [d0, ..., d(level), ACTION_DIM]
        N = level_config[0]
        dm = distance_matrix(euclidian_distance, subdata, axis=level) # shape [dim0, dim1, ..., dim(level), dim(level)]
        # 1e-3 because of the small epsilon in the euclidian_distance function definition
        dm = np.ma.masked_equal(dm, 1e-3) # shape [dim0, dim1, ..., dim(level), dim(level)]
        min_dist = np.min(dm, axis=level)
        # return np.sort(min_dist, axis=-1)
        return min_dist

    return plot_hierarchy(
        recomputed_returns,
        hierarchization_config,
        func=func,
    )


def plot_max_distance_hierarchy(recomputed_returns, hierarchization_config, cmap='hot'):
    def func(subdata, level, level_config):
        # subdata shape: [d0, ..., d(level), ACTION_DIM]
        N = level_config[0]
        dm = distance_matrix(euclidian_distance, subdata, axis=level) # shape [dim0, dim1, ..., dim(level), dim(level)]
        # 1e-3 because of the small epsilon in the euclidian_distance function definition
        dm = np.ma.masked_equal(dm, 1e-3) # shape [dim0, dim1, ..., dim(level), dim(level)]
        max_dist = np.max(dm, axis=level)
        # return np.sort(max_dist, axis=-1)
        return max_dist

    return plot_hierarchy(
        recomputed_returns,
        hierarchization_config,
        func=func,
    )


if __name__ == '__main__':


    k = 1.15
    l = 2
    dmin = 0.8
    dmax = dmin * l ** 0.6
    hierarchization_config = (
        (7, dmin * l ** 2, dmax * l ** 2, 1 / k ** 0, 1 / k ** 0),
        (7, dmin * l ** 1, dmax * l ** 1, 1 / k ** 1, 1 / k ** 1),
        (7, dmin * l ** 0, dmax * l ** 0, 1 / k ** 2, 1 / k ** 2),
    )
    N_ACTORS = int(np.prod(tuple(dim for dim, _, _, _, _ in hierarchization_config)))

    recomputed_returns = np.random.uniform(size=(N_ACTORS,), high=12)
    recomputed_actions = np.random.uniform(size=(N_ACTORS, 7), low=-1, high=1)


    plot_returns_hierarchy(recomputed_returns, hierarchization_config)
    plt.show()

    # plot_loss_hierarchy(recomputed_actions, hierarchization_config)
    # plt.show()

    # plot_min_distance_hierarchy(recomputed_actions, hierarchization_config)
    # plt.show()
