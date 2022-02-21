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
# level=0 - dim=45 - d_min=2.54 - d_max=3.81 - slope_min=0.86 - slope_max=0.86
# level=1 - dim=4 - d_min=0.6 - d_max=0.90 - slope_min=0.75 - slope_max=0.75


defaults = {
    "hierarchisation_args": {
        "n_actors": 180,
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


improvement_1 = {
    "hierarchisation_args": {
        "n_actors": 180,
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
        "n_expl_ep_per_it": 160, # changed
        "n_nonexpl_ep_per_it": 0, # changed
        "experiment_length_in_ep":16000 ,
        "n_actor_pretraining": 0,
        "n_critic_training_per_loop_iteration": 800, # changed
        "n_actor_training_per_loop_iteration": 100,
    },

    "hyperparameters_config_args": {
        "discount_factor": 0.9,
        "noise_magnitude_limit": 2.0, # changed
        "hierarchization_coef": 0.01, # changed
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


def exp_2022_18_02_search_noise_magnitude_limit(db):
    args = deepcopy(defaults)
    for noise_magnitude_limit in [0.01, 0.25, 0.5, 1.0, 1.5, 2.0]:
        args["hyperparameters_config_args"]["noise_magnitude_limit"] = noise_magnitude_limit
        insert_args(db, args, ["parameter_search", "search_noise_magnitude_limit"])


def exp_2022_18_02_search_expl_vs_nonexpl(db):
    args = deepcopy(defaults)
    for n_nonexpl_ep_per_it in [0, 40, 80, 100]:
        n_expl_ep_per_it = 160 - n_nonexpl_ep_per_it
        args["mainloop_config_args"]["n_nonexpl_ep_per_it"] = n_nonexpl_ep_per_it
        args["mainloop_config_args"]["n_expl_ep_per_it"] = n_expl_ep_per_it
        insert_args(db, args, ["parameter_search", "search_expl_vs_nonexpl"])


def exp_2022_19_02_search_critic_lr_and_n_training(db):
    args = deepcopy(defaults)
    for nc, bs, lr in [(200, 8, 1e-3), (200, 8, 5e-3), (200, 8, 25e-3), (400, 4, 1e-3), (400, 4, 5e-3), (400, 4, 25e-3)]:
        args["mainloop_config_args"]["n_critic_training_per_loop_iteration"] = nc
        args["hyperparameters_config_args"]["batch_size"] = bs
        args["hyperparameters_config_args"]["critic_learning_rate"] = lr
        insert_args(db, args, ["parameter_search", "search_critic_lr_and_n_training"])


def exp_2022_19_02_search_actor_lr_and_n_training(db):
    args = deepcopy(defaults)
    for na, bs, lr in [(50, 8, 5e-6), (50, 8, 2e-5), (50, 8, 1e-4), (100, 8, 2e-5), (100, 8, 5e-6), (100, 4, 2e-5)]:
        args["mainloop_config_args"]["n_actor_training_per_loop_iteration"] = na
        args["hyperparameters_config_args"]["batch_size"] = bs
        args["hyperparameters_config_args"]["actor_learning_rate"] = lr
        insert_args(db, args, ["parameter_search", "search_actor_lr_and_n_training"])


def exp_2022_19_02_search_discount_factor(db):
    args = deepcopy(defaults)
    for df in [0.7, 0.8, 0.9, 0.95]:
        args["hyperparameters_config_args"]["discount_factor"] = df
        insert_args(db, args, ["parameter_search", "search_discount_factor"])


def exp_2022_20_02_search_shorter_loop(db):
    args = deepcopy(improvement_1)
    args["mainloop_config_args"]["n_nonexpl_ep_per_it"] = 0
    for n_expl_ep_per_it in [20, 40, 80, 160]:
        args["mainloop_config_args"]["n_expl_ep_per_it"] = n_expl_ep_per_it
        args["mainloop_config_args"]["n_critic_training_per_loop_iteration"] = 5 * n_expl_ep_per_it
        args["mainloop_config_args"]["n_actor_training_per_loop_iteration"] = int(n_expl_ep_per_it * 100 / 160)
        args["mainloop_config_args"]["lookback"] = 4 * 160 // n_expl_ep_per_it
        insert_args(db, args, ["parameter_search", "search_shorter_loop"])


def exp_2022_20_02_search_better_hierarchization(db):
    args = deepcopy(improvement_1)

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
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])

    hierarchization_config = (
        (20, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (9, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])

    hierarchization_config = (
        (9, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (20, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])

    hierarchization_config = (
        (4, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (45, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])


    k = 1.15
    SQRT2 = 1.41421356237
    SAFETY = 1.15
    minmax_factor = 1.5
    dmin2 = 0.4
    dmax2 = dmin2 * minmax_factor
    dmin1 = SAFETY * SQRT2 * (dmax2)
    dmax1 = dmin1 * minmax_factor
    dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
    dmax0 = dmin0 * minmax_factor


    hierarchization_config = (
        (5, dmin0, dmax0, 1 / k ** 1, 1 / k ** 0),
        (6, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (6, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])


    hierarchization_config = (
        (4, dmin0, dmax0, 1 / k ** 1, 1 / k ** 0),
        (5, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (9, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])


    hierarchization_config = (
        (9, dmin0, dmax0, 1 / k ** 1, 1 / k ** 0),
        (5, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
        (4, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization"])


def exp_2022_20_02_search_better_hierarchization_2(db):
    args = deepcopy(improvement_1)

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
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])

    hierarchization_config = (
        (20, dmin1, 100, 1 / k ** 1, 1 / k ** 1),
        (9, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])

    hierarchization_config = (
        (30, dmin1, 100, 1 / k ** 1, 1 / k ** 1),
        (6, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])

    hierarchization_config = (
        (36, dmin1, 100, 1 / k ** 1, 1 / k ** 1),
        (5, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])

    hierarchization_config = (
        (45, dmin1, 100, 1 / k ** 1, 1 / k ** 1),
        (4, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])

    hierarchization_config = (
        (60, dmin1, 100, 1 / k ** 1, 1 / k ** 1),
        (3, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])

    hierarchization_config = (
        (90, dmin1, 100, 1 / k ** 1, 1 / k ** 1),
        (2, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
    )
    args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
    insert_args(db, args, ["parameter_search", "search_better_hierarchization_no_dmax1"])


    for dmin2 in [0.2, 0.4, 0.6, 0.8]:
        k = 1.15
        SQRT2 = 1.41421356237
        SAFETY = 2
        minmax_factor = 1.5
        dmax2 = dmin2 * minmax_factor
        dmin1 = SAFETY * SQRT2 * (dmax2)
        dmax1 = dmin1 * minmax_factor
        dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
        dmax0 = dmin0 * minmax_factor

        hierarchization_config = (
            (45, dmin1, dmax1, 1 / k ** 1, 1 / k ** 1),
            (4, dmin2, dmax2, 1 / k ** 2, 1 / k ** 2),
        )
        args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
        insert_args(db, args, ["parameter_search", "search_better_hierarchization_dmin2"])

    for minmax_factor in [1.2, 1.5, 1.7]:
        k = 1.15
        SQRT2 = 1.41421356237
        SAFETY = 2
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
        args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
        insert_args(db, args, ["parameter_search", "search_better_hierarchization_minmax_factor"])


    for SAFETY in [1.2, 1.5, 1.7, 2]:
        k = 1.15
        SQRT2 = 1.41421356237
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
        args["hierarchisation_args"]["hierarchization_config"] = hierarchization_config
        insert_args(db, args, ["parameter_search", "search_better_hierarchization_safety"])








experiment_sets = {
    "exp_2022_16_02_search_hierarchization_coef": exp_2022_16_02_search_hierarchization_coef,
    "exp_2022_16_02_search_critic_training_per_loop": exp_2022_16_02_search_critic_training_per_loop,
    "exp_2022_16_02_search_batch_size": exp_2022_16_02_search_batch_size,
    "exp_2022_16_02_search_lookback": exp_2022_16_02_search_lookback,
    "exp_2022_18_02_search_actor_training_per_loop": exp_2022_18_02_search_actor_training_per_loop,
    "exp_2022_18_02_search_noise_magnitude_limit": exp_2022_18_02_search_noise_magnitude_limit,
    "exp_2022_18_02_search_expl_vs_nonexpl": exp_2022_18_02_search_expl_vs_nonexpl,
    "exp_2022_19_02_search_critic_lr_and_n_training": exp_2022_19_02_search_critic_lr_and_n_training,
    "exp_2022_19_02_search_actor_lr_and_n_training": exp_2022_19_02_search_actor_lr_and_n_training,
    "exp_2022_19_02_search_discount_factor": exp_2022_19_02_search_discount_factor,
    "exp_2022_20_02_search_shorter_loop": exp_2022_20_02_search_shorter_loop,
    "exp_2022_20_02_search_better_hierarchization": exp_2022_20_02_search_better_hierarchization,
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
    parser.add_argument('--db-name', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')
    parser.add_argument('set_names', nargs='+', help='names of the sets of experiments to add to the DB')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    for set_name in args.set_names:
        experiment_sets[set_name](db)

    # exp_2022_16_02_search_hierarchization_coef exp_2022_16_02_search_critic_training_per_loop exp_2022_16_02_search_batch_size exp_2022_16_02_search_lookback
