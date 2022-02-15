from datetime import datetime
import sqlite3 as sql
import pandas as pd
import sys
import logging
import pickle
from contextlib import contextmanager
from collections import namedtuple
from exploration import ExplorationConfig


ACTION_DIM = 7


Args = namedtuple("Args", ["agent", "experiment", "mainloop"])


class Database:
    def __init__(self, path):
        self._logger = logging.getLogger("Database")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info(f"[database] opening {path}")
        self.path = path
        self.conn = sql.connect(path, detect_types=sql.PARSE_DECLTYPES)
        self.conn.set_trace_callback(self._logger.debug)
        self.cursor = self.conn.cursor()
        command = 'PRAGMA foreign_keys = ON;'
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS exploration_configs (
                     exploration_config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     type TEXT NOT NULL,
                     N INTEGER,
                     interpolation_type TEXT,
                     upsilon_t0 FLOAT,
                     upsilon_tN FLOAT,
                     exploration_prob_t0 FLOAT,
                     exploration_prob_tN FLOAT,
                     softmax_temperature_t0 FLOAT,
                     softmax_temperature_tN FLOAT,
                     CONSTRAINT UC_exploration_config UNIQUE (
                        type,
                        N,
                        interpolation_type,
                        upsilon_t0,
                        upsilon_tN,
                        exploration_prob_t0,
                        exploration_prob_tN,
                        softmax_temperature_t0,
                        softmax_temperature_tN
                    )
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS hierarchization_configs (
                     hierarchization_config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     n_actors INTEGER NOT NULL,
                     hierarchization_config VARBINARY(1024),
                     CONSTRAINT UC_hierarchization_config UNIQUE (hierarchization_config)
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS experiment_configs (
                     experiment_config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     hierarchization_config_id INTEGER NOT NULL,
                     exploration_config_id INTEGER NOT NULL,
                     repetitions_total INTEGER NOT NULL,
                     repetitions_remaining INTEGER NOT NULL,
                     repetitions_running INTEGER NOT NULL,
                     repetitions_done INTEGER NOT NULL,
                     restore_path TEXT,
                     n_sim INTEGER NOT NULL,
                     discount_factor FLOAT NOT NULL,
                     noise_magnitude_limit FLOAT NOT NULL,
                     hierarchization_coef FLOAT NOT NULL,
                     actor_learning_rate FLOAT NOT NULL,
                     critic_learning_rate FLOAT NOT NULL,
                     batch_size INTEGER NOT NULL,
                     episode_length INTEGER NOT NULL,
                     lookback INTEGER NOT NULL,
                     smoothing FLOAT NOT NULL,
                     n_expl_ep_per_it INTEGER NOT NULL,
                     n_nonexpl_ep_per_it INTEGER NOT NULL,
                     experiment_length_in_ep INTEGER NOT NULL,
                     n_critic_training_per_loop_iteration INTEGER NOT NULL,
                     n_actor_training_per_loop_iteration INTEGER NOT NULL,

                     FOREIGN KEY (hierarchization_config_id)
                        REFERENCES hierarchization_configs(hierarchization_config_id)
                        ON DELETE CASCADE,
                     FOREIGN KEY (exploration_config_id)
                        REFERENCES exploration_configs(exploration_config_id)
                        ON DELETE CASCADE,
                     CONSTRAINT UC_experiment_config UNIQUE (
                        restore_path,
                        hierarchization_config_id,
                        exploration_config_id,
                        discount_factor,
                        noise_magnitude_limit,
                        hierarchization_coef,
                        actor_learning_rate,
                        critic_learning_rate,
                        batch_size,
                        episode_length,
                        lookback,
                        smoothing,
                        n_expl_ep_per_it,
                        n_nonexpl_ep_per_it,
                        experiment_length_in_ep,
                        n_critic_training_per_loop_iteration,
                        n_actor_training_per_loop_iteration
                     ),
                     CONSTRAINT CC_divisible_n_sim CHECK (n_expl_ep_per_it % n_sim = 0 AND n_nonexpl_ep_per_it % n_sim = 0),
                     CONSTRAINT CC_divisible_n_ep CHECK (experiment_length_in_ep % (n_expl_ep_per_it + n_nonexpl_ep_per_it) = 0)
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS experiments (
                     experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     experiment_config_id INTEGER NOT NULL,
                     PRNGKey_start INTEGER NOT NULL,
                     date_time_start DATETIME NOT NULL,
                     date_time_stop DATETIME,
                     hourly_pricing FLOAT,
                     path TEXT NOT NULL,
                     finished INTEGER NOT NULL DEFAULT 0,
                     FOREIGN KEY (experiment_config_id)
                        REFERENCES experiment_configs(experiment_config_id)
                        ON DELETE CASCADE
                    CONSTRAINT UC_PRNGKey UNIQUE (experiment_config_id, PRNGKey_start)
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS results (
                     experiment_id INTEGER NOT NULL,
                     loop_iteration INTEGER NOT NULL,
                     episode_nb INTEGER NOT NULL,
                     training_episode_return FLOAT,
                     testing_episode_return FLOAT,
                     exploration_param FLOAT NOT NULL,
                     FOREIGN KEY (experiment_id)
                        REFERENCES experiments(experiment_id)
                        ON DELETE CASCADE
                     CONSTRAINT UC_loop_iteration UNIQUE (experiment_id,loop_iteration),
                     CONSTRAINT UC_episode_nb UNIQUE (experiment_id,episode_nb)
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS collections (
                     collection TEXT PRIMARY KEY
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS collections_experiment_config (
                     collection TEXT NOT NULL,
                     experiment_config_id INTEGER NOT NULL,
                     FOREIGN KEY (collection)
                        REFERENCES collections(collection)
                     FOREIGN KEY (experiment_config_id)
                        REFERENCES experiment_configs(experiment_config_id)
                        ON DELETE CASCADE
                     UNIQUE (collection, experiment_config_id)
                  );'''
        self.cursor.execute(command)

    def add_to_collection(self, experiment_config_id, collection, or_ignore=False):
        self.insert_into("collections", {
            "collection": collection
        }, or_ignore=or_ignore)
        self.insert_into("collections_experiment_config", {
            "collection": collection,
            "experiment_config_id": experiment_config_id
        }, or_ignore=or_ignore)

    def insert_experiment_config(self,
            hierarchization_config_id, exploration_config_id, repetitions_total, restore_path, n_sim,
            discount_factor, noise_magnitude_limit, hierarchization_coef,
            actor_learning_rate, critic_learning_rate, batch_size,
            episode_length, lookback, smoothing, n_expl_ep_per_it, n_nonexpl_ep_per_it,
            experiment_length_in_ep, n_critic_training_per_loop_iteration,
            n_actor_training_per_loop_iteration, or_ignore=False):
        return self.insert_into("experiment_configs", {
            "hierarchization_config_id": hierarchization_config_id,
            "exploration_config_id": exploration_config_id,
            "repetitions_total": repetitions_total,
            "repetitions_remaining": repetitions_total,
            "repetitions_running": 0,
            "repetitions_done": 0,
            "restore_path": restore_path,
            "n_sim": n_sim,
            "discount_factor": discount_factor,
            "noise_magnitude_limit": noise_magnitude_limit,
            "hierarchization_coef": hierarchization_coef,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "batch_size": batch_size,
            "episode_length": episode_length,
            "lookback": lookback,
            "smoothing": smoothing,
            "n_expl_ep_per_it": n_expl_ep_per_it,
            "n_nonexpl_ep_per_it": n_nonexpl_ep_per_it,
            "experiment_length_in_ep": experiment_length_in_ep,
            "n_critic_training_per_loop_iteration": n_critic_training_per_loop_iteration,
            "n_actor_training_per_loop_iteration": n_actor_training_per_loop_iteration,
        }, or_ignore=or_ignore)

    def insert_experiment(self, experiment_config_id, PRNGKey_start, date_time_start,
            hourly_pricing, path, or_ignore=False):
        return self.insert_into("experiments", {
            "experiment_config_id": experiment_config_id,
            "PRNGKey_start": PRNGKey_start,
            "date_time_start": date_time_start,
            "hourly_pricing": hourly_pricing,
            "path": path,
        }, or_ignore=or_ignore)

    def insert_exploration_config(self, type, N, interpolation_type, upsilon_t0,
            upsilon_tN, exploration_prob_t0, exploration_prob_tN,
            softmax_temperature_t0, softmax_temperature_tN, or_ignore=False):
        return self.insert_into("exploration_configs", {
            "type": type,
            "N": N,
            "interpolation_type": interpolation_type,
            "upsilon_t0": upsilon_t0,
            "upsilon_tN": upsilon_tN,
            "exploration_prob_t0": exploration_prob_t0,
            "exploration_prob_tN": exploration_prob_tN,
            "softmax_temperature_t0": softmax_temperature_t0,
            "softmax_temperature_tN": softmax_temperature_tN,
        }, or_ignore=or_ignore)

    def insert_hierarchization_config(self, n_actors, hierarchization_config, or_ignore=False):
        return self.insert_into("hierarchization_configs", {
            "n_actors": n_actors,
            "hierarchization_config": pickle.dumps(hierarchization_config),
        }, or_ignore=or_ignore)

    def insert_result(self, experiment_id, loop_iteration, episode_nb,
        training_episode_return, testing_episode_return, exploration_param, or_ignore=False):
        self.insert_into("results", {
            "experiment_id": experiment_id,
            "loop_iteration": loop_iteration,
            "episode_nb": episode_nb,
            "training_episode_return": training_episode_return,
            "testing_episode_return": testing_episode_return,
            "exploration_param": exploration_param,
        }, or_ignore=or_ignore)

    def insert_into(self, table_name, name_values_dict, or_ignore=False):
        OR_IGNORE = "OR IGNOER" if or_ignore else ""
        command = ""
        command += f"INSERT {OR_IGNORE} INTO {table_name}(\n    "
        command += ",\n    ".join(name_values_dict.keys())
        command += f"\n) VALUES ({','.join(['?'] * len(name_values_dict))})"
        command_get_id = "SELECT last_insert_rowid()"
        with self.conn:
            self.cursor.execute(command, tuple(name_values_dict.values()))
            self.conn.commit()
            self.cursor.execute(command_get_id)
            return self.cursor.fetchone()[0]

    def get_dataframe(self, command):
        return pd.read_sql(command, self.conn)

    def get_experiment_config_ids(self, **kwargs):
        where = tuple(f"{key} IS NULL" if value is None else f"{key}='{value}'" for key, value in kwargs.items())
        where = ' AND\n            '.join(where)
        command = 'SELECT experiment_config_id FROM experiment_configs WHERE' + f'\n{where}' if kwargs else ''
        self.cursor.execute(command)
        return self.cursor.fetchall()

    def get_experiment_config_field(self, experiment_config_id, field):
        command = f"SELECT {field} FROM experiment_configs WHERE experiment_config_id={experiment_config_id}"
        self.cursor.execute(command)
        return self.cursor.fetchall()

    def delete_hierarchization_config(self, hierarchization_config_id):
        self._logger.info(f'[database] deleting hierarchization_config with id {hierarchization_config_id}')
        command = f'DELETE FROM hierarchization_configs WHERE hierarchization_config_id={hierarchization_config_id}'
        self.cursor.execute(command)
        self.conn.commit()

    def delete_experiment_config(self, experiment_config_id):
        self._logger.info(f'[database] deleting experiment_config with id {experiment_config_id}')
        command = f'DELETE FROM experiment_configs WHERE experiment_config_id={experiment_config_id}'
        self.cursor.execute(command)
        self.conn.commit()

    def delete_experiment(self, experiment_id):
        self._logger.info(f'[database] deleting experiment with id {experiment_id}')
        command = f'DELETE FROM experiments WHERE experiment_id={experiment_id}'
        self.cursor.execute(command)
        self.conn.commit()

    def register_termination(self, experiment_id, date_time_stop):
        command = f"UPDATE experiments SET finished=1,date_time_stop='{date_time_stop}' WHERE experiment_id={experiment_id}"
        self.cursor.execute(command)
        self.conn.commit()

    def get_hierarchization_config(self, hierarchization_config_id):
        command = f'SELECT hierarchization_config FROM hierarchization_configs WHERE hierarchization_config_id={hierarchization_config_id}'
        self.cursor.execute(command)
        data = self.cursor.fetchone()[0]
        return pickle.loads(data)

    def get_PRNGKey_start(self, experiment_config_id):
        command = f'SELECT PRNGKey_start FROM experiments WHERE experiment_config_id={experiment_config_id}'
        self.cursor.execute(command)
        PRNGKeys = tuple(x[0] for x in self.cursor.fetchall())
        key = 0
        while key in PRNGKeys:
            key += 1
        return key

    def get_exploration_config(self, exploration_config_id):
        command = f'SELECT * FROM exploration_configs WHERE exploration_config_id={exploration_config_id}'
        self.cursor.execute(command)
        args = self.cursor.fetchone()[1:]
        return ExplorationConfig(*args)

    @contextmanager
    def get_a_job(self, path):
        with self.conn:
            command = f'SELECT * FROM experiment_configs WHERE repetitions_remaining > 0'
            self.cursor.execute(command)
            res = self.cursor.fetchone() # tuple containing the data (one row)
            if res is None:
                yield None
                return
            (
                experiment_config_id,
                hierarchization_config_id,
                exploration_config_id,
                repetitions_total,
                repetitions_remaining,
                repetitions_running,
                repetitions_done,
                restore_path,
                n_sim,
                discount_factor,
                noise_magnitude_limit,
                hierarchization_coef,
                actor_learning_rate,
                critic_learning_rate,
                batch_size,
                episode_length,
                lookback,
                smoothing,
                n_expl_ep_per_it,
                n_nonexpl_ep_per_it,
                experiment_length_in_ep,
                n_critic_training_per_loop_iteration,
                n_actor_training_per_loop_iteration,
            ) = res
            command = f'''UPDATE experiment_configs
                          SET
                            repetitions_remaining = repetitions_remaining - 1,
                            repetitions_running = repetitions_running + 1
                          WHERE
                            experiment_config_id={experiment_config_id}
                          '''
            self.cursor.execute(command)
            PRNGKey_start = self.get_PRNGKey_start(experiment_config_id)
            experiment_id = self.insert_experiment(experiment_config_id, PRNGKey_start, datetime.now(), 1.35, path)
        hierarchization_config = self.get_hierarchization_config(hierarchization_config_id)
        exploration_config = self.get_exploration_config(exploration_config_id)
        agent_args = (
            discount_factor,
            noise_magnitude_limit,
            hierarchization_config,
            hierarchization_coef,
            actor_learning_rate,
            critic_learning_rate,
            ACTION_DIM
        )
        experiment_args = (n_sim, batch_size, episode_length)
        mainloop_args = (
            PRNGKey_start,
            lookback,
            n_expl_ep_per_it,
            n_nonexpl_ep_per_it,
            experiment_length_in_ep,
            n_critic_training_per_loop_iteration,
            n_actor_training_per_loop_iteration,
            exploration_config,
            repetitions_remaining == repetitions_total,
            restore_path,
            path,
            self,
            experiment_id,
        )
        try:
            yield Args(agent_args, experiment_args, mainloop_args)
        except Exception as e:
            with self.conn:
                self._logger.critical("An exception has occured, deleting the experiment {experiment_id}")
                self.delete_experiment(experiment_id)
                self._logger.critical("An exception has occured, incrementing the 'remaining' counter")
                command = f'''UPDATE experiment_configs
                              SET
                                repetitions_remaining = repetitions_remaining + 1
                              WHERE
                                experiment_config_id={experiment_config_id}
                              '''
                self.cursor.execute(command)
                raise e
        finally:
            with self.conn:
                command = f'''UPDATE experiment_configs
                              SET
                                repetitions_running = repetitions_running - 1
                              WHERE
                                experiment_config_id={experiment_config_id}
                              '''
                self.cursor.execute(command)
        self.register_termination(experiment_id, datetime.now())


if __name__ == '__main__':
    db = Database('/tmp/debug.db')

    n_actors = 123
    hierarchization_config = [(1, 2, 3), (4, 5, 6)]
    type = "softmax_temperature"
    N = 10000
    interpolation_type = 'cosine'
    upsilon_t0 = None
    upsilon_tN = None
    exploration_prob_t0 = None
    exploration_prob_tN = None
    softmax_temperature_t0 = 1.0
    softmax_temperature_tN = 4.0
    repetitions_total = 10
    restore_path = None
    n_sim = 20
    discount_factor = 0.9
    noise_magnitude_limit = 0.5
    hierarchization_coef = 0.1
    actor_learning_rate = 5e-4
    critic_learning_rate = 1e-3
    batch_size = 4
    exploration_config_id = 0.5
    episode_length = 100
    lookback = 4
    smoothing = 0.0
    n_expl_ep_per_it = 80
    n_nonexpl_ep_per_it = 80
    experiment_length_in_ep = 32000
    n_critic_training_per_loop_iteration = 400
    n_actor_training_per_loop_iteration = 100
    collection = "debug_collection"
    date_time_start = datetime.now()
    PRNGKey_start = 0
    hourly_pricing = 1.35
    path = "/some/path/to/exp"
    loop_iteration = 0
    episode_nb = 0
    training_episode_return = 2.0
    testing_episode_return = 2.0
    exploration_param = 0.4

    hierarchization_config_id = db.insert_hierarchization_config(
        n_actors,
        hierarchization_config,
    )

    exploration_config_id = db.insert_exploration_config(
        type,
        N,
        interpolation_type,
        upsilon_t0,
        upsilon_tN,
        exploration_prob_t0,
        exploration_prob_tN,
        softmax_temperature_t0,
        softmax_temperature_tN,
    )

    experiment_config_id = db.insert_experiment_config(
        hierarchization_config_id,
        exploration_config_id,
        repetitions_total,
        restore_path,
        n_sim,
        discount_factor,
        noise_magnitude_limit,
        hierarchization_coef,
        actor_learning_rate,
        critic_learning_rate,
        batch_size,
        episode_length,
        lookback,
        smoothing,
        n_expl_ep_per_it,
        n_nonexpl_ep_per_it,
        experiment_length_in_ep,
        n_critic_training_per_loop_iteration,
        n_actor_training_per_loop_iteration,
    )

    db.add_to_collection(experiment_config_id, collection)

    experiment_id = db.insert_experiment(
        experiment_config_id,
        PRNGKey_start,
        date_time_start,
        hourly_pricing,
        path,
    )

    db.insert_result(
        experiment_id,
        loop_iteration,
        episode_nb,
        training_episode_return,
        testing_episode_return,
        exploration_param,
    )

    def print_db(db):
        print("####################################################################")
        print(db.get_dataframe("SELECT * FROM experiment_configs"))
        print("\n\n")
        print(db.get_dataframe("SELECT * FROM hierarchization_configs"))
        print("\n\n")
        print(db.get_dataframe("SELECT * FROM exploration_configs"))
        print("\n\n")
        print(db.get_dataframe("SELECT * FROM experiments"))
        print("\n\n")
        print(db.get_dataframe("SELECT * FROM results"))
        print("\n\n")
        print(db.get_dataframe("SELECT * FROM collections_experiment_config"))
        print("\n\n")
        print(db.get_dataframe("SELECT * FROM collections"))
        print("####################################################################")


    print_db(db)

    # db.delete_experiment_config(experiment_config_id)
    # db.delete_experiment(experiment_id)

    print_db(db)
