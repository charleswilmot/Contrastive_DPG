from database import Database
from copy import deepcopy


k = 1.15
SQRT2 = 1.41421356237
SAFETY = 2
minmax_factor = 1.5
dmin2 = 0.6
dmax2 = dmin2 * minmax_factor
dmin1 = SAFETY * SQRT2 * (dmax2)
dmax1 = dmin1 * minmax_factor
dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
dmax0 = dmin0 * minmax_factor


hierarchization_config = (
    (45, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
    (4, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
)
# level=0 - dim=60 - d_min=2.54 - d_max=3.81 - slope_min=0.86 - slope_max=0.86
# level=1 - dim=3 - d_min=0.6 - d_max=0.90 - slope_min=0.75 - slope_max=0.75


defaults = {
    "hierarchisation_args": {
        "n_actors": 123,
        "hierarchization_config": hierarchization_config,
    },

    "exploration_config_args": {
        "type": "exploration_prob",
        "N": 6000,
        "interpolation_type": 'cosine',
        "upsilon_t0": 0.2,
        "upsilon_tN": 0.6,
        "exploration_prob_t0": 0.9,
        "exploration_prob_tN": 0.05,
        "softmax_temperature_t0": 1.0,
        "softmax_temperature_tN": 0.25,
    },

    "mainloop_config_args": {
        "restore_path": "",
        "n_sim": 20,
        "episode_length": 100,
        "lookback": 4,
        "n_expl_ep_per_it": 120,
        "n_nonexpl_ep_per_it": 40,
        "experiment_length_in_ep":16000 ,
        "n_actor_pretraining": 0,
        "n_critic_training_per_loop_iteration": 400,
        "n_actor_training_per_loop_iteration": 100,
    },

    "hyperparameters_config_args": {
        "discount_factor": 0.9,
        "noise_magnitude_limit": 0.5,
        "hierarchization_coef": 0.1,
        "actor_learning_rate": 2e-5,
        "critic_learning_rate": 1e-3,
        "batch_size": 4,
        "smoothing": 0.0,
    },

    "experiment_config_args": {
        "repetitions_total": 1,
    },
}


def insert_args(db, args, collections):
    hierarchization_config_id = db.insert_hierarchization_config(
        **args["hierarchisation_args"],
        protect=False,
    )
    exploration_config_id = db.insert_exploration_config(
        **args["exploration_config_args"],
        protect=False,
    )
    mainloop_config_id = db.insert_mainloop_config(
        **args["mainloop_config_args"],
        protect=False,
    )
    hyperparameters_config_id = db.insert_hyperparameters_config(
        hierarchization_config_id,
        exploration_config_id,
        **args["hyperparameters_config_args"],
        protect=False,
    )
    experiment_config_id = db.insert_experiment_config(
        mainloop_config_id,
        hyperparameters_config_id,
        **args["experiment_config_args"],
        protect=False,
    )
    for collection in collections:
        db.add_to_collection(
            experiment_config_id,
            collection,
            protect=False,
        )



def exp_2022_16_02_search_hierarchization_coef(db):
    args = deepcopy(defaults)
    for hierarchization_coef in [0.01, 0.1, 1.0, 10.0]:
        args["hyperparameters_config_args"]["hierarchization_coef"] = hierarchization_coef
        insert_args(db, args, ["parameter_search", "search_hierarchization_coef"])


def exp_2022_16_02_search_critic_training_per_loop(db):
    args = deepcopy(defaults)
    for n_critic_training_per_loop_iteration in [100, 200, 400, 800, 1600]:
        args["mainloop_config_args"]["n_critic_training_per_loop_iteration"] = n_critic_training_per_loop_iteration
        insert_args(db, args, ["parameter_search", "search_critic_training_per_loop"])


def exp_2022_16_02_search_batch_size(db):
    args = deepcopy(defaults)
    for batch_size in [2, 4, 8, 16]:
        args["hyperparameters_config_args"]["batch_size"] = batch_size
        insert_args(db, args, ["parameter_search", "search_batch_size"])


def exp_2022_16_02_search_lookback(db):
    args = deepcopy(defaults)
    for lookback in [2, 4, 8]:
        args["mainloop_config_args"]["lookback"] = lookback
        insert_args(db, args, ["parameter_search", "search_lookback"])


def exp_2022_18_02_search_actor_training_per_loop(db):
    args = deepcopy(defaults)
    for n_actor_training_per_loop_iteration in [10, 50, 100, 200, 400]:
        args["mainloop_config_args"]["n_actor_training_per_loop_iteration"] = n_actor_training_per_loop_iteration
        insert_args(db, args, ["parameter_search", "search_actor_training_per_loop"])


experiment_sets = {
    "exp_2022_16_02_search_hierarchization_coef": exp_2022_16_02_search_hierarchization_coef,
    "exp_2022_16_02_search_critic_training_per_loop": exp_2022_16_02_search_critic_training_per_loop,
    "exp_2022_16_02_search_batch_size": exp_2022_16_02_search_batch_size,
    "exp_2022_16_02_search_lookback": exp_2022_16_02_search_lookback,
    "exp_2022_18_02_search_actor_training_per_loop": exp_2022_18_02_search_actor_training_per_loop,
}


if __name__ == '__main__':
    import argparse
    import logging
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Populate database with experiments.')
    parser.add_argument('--user', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DPG_debug', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')
    parser.add_argument('set_names', nargs='+', help='names of the sets of experiments to add to the DB')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    for set_name in args.set_names:
        experiment_sets[set_name](db)

    # exp_2022_16_02_search_hierarchization_coef exp_2022_16_02_search_critic_training_per_loop exp_2022_16_02_search_batch_size exp_2022_16_02_search_lookback
