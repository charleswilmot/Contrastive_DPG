import sys
sys.path.insert(1, '../src')
from distance_matrix import *
from timeit import timeit


NUMBER = 10

N_ACTORS = 200
BATCH = 2
SEQUENCE = 100
ACTION_DIM = 7

x = jnp.zeros(shape=(N_ACTORS, BATCH, SEQUENCE, ACTION_DIM))


def profile(func, string):
    func()
    t = timeit(func, number=NUMBER)
    print(f'{string}: {t}')


profile(
    func=lambda: distance_matrix(euclidian_distance, x),
    string="distance_matrix",
)
profile(
    func=lambda: n_closest(euclidian_distance, x, n=5),
    string=f"n_closest (euclidian_distance)",
)
profile(
    func=lambda: n_closest(squared_distance, x, n=5),
    string=f"n_closest (squared_distance)",
)
