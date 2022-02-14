import numpy as np
import jax.numpy as jnp


def reshape(data, hierarchization_config):
    dims = tuple(dim for dim, _, _, _, _ in hierarchization_config)
    if len(data) != int(np.prod(dims)):
        raise RuntimeError(f"The number of points does not match with the description ({data.shape=} {hierarchization_config=})")
    shape = (*dims, *data.shape[1:]) # [dim0, dim1, ..., dimn, ...A...]
    return jnp.reshape(data, shape) # [dim0, dim1, ..., dimn, ...A...]


def map(data, hierarchization_config, func=lambda subdata, level, level_config: subdata):
    reshaped_data = reshape(data, hierarchization_config)
    n_levels = len(hierarchization_config)
    ret = []
    for level, level_config in enumerate(hierarchization_config):
        indices = tuple(slice(None) if i <= level else 0 for i in range(n_levels))
        subdata = reshaped_data[indices] # shape [dim0, ..., dim(l), ...A...]
        ret.append(func(subdata, level, level_config))
    return ret
