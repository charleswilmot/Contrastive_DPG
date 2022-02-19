from datetime import datetime
import mysql.connector as sql
from mysql.connector import errorcode
import pandas as pd
import sys
import logging
import pickle
from contextlib import contextmanager
from collections import namedtuple
from exploration import ExplorationConfig


ACTION_DIM = 7


Args = namedtuple("Args", ["agent", "experiment", "mainloop"])


TABLES = {
    "exploration_configs": '''
        CREATE TABLE IF NOT EXISTS exploration_configs (
            exploration_config_id INTEGER PRIMARY KEY AUTO_INCREMENT,
            type VARCHAR(255) NOT NULL,
            N INTEGER,
            interpolation_type VARCHAR(255),
            upsilon_t0 DECIMAL(12,7),
            upsilon_tN DECIMAL(12,7),
            exploration_prob_t0 DECIMAL(12,7),
            exploration_prob_tN DECIMAL(12,7),
            softmax_temperature_t0 DECIMAL(12,7),
            softmax_temperature_tN DECIMAL(12,7),
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
        );''',

    "hierarchization_configs": '''
        CREATE TABLE IF NOT EXISTS hierarchization_configs (
            hierarchization_config_id INTEGER PRIMARY KEY AUTO_INCREMENT,
            n_actors INTEGER NOT NULL,
            hierarchization_config VARBINARY(1024),
            CONSTRAINT UC_hierarchization_config UNIQUE (hierarchization_config)
        );''',

    "mainloop_configs": '''
        CREATE TABLE IF NOT EXISTS mainloop_configs (
            mainloop_config_id INTEGER PRIMARY KEY AUTO_INCREMENT,
            restore_path VARCHAR(255),
            n_sim INTEGER NOT NULL,
            episode_length INTEGER NOT NULL,
            lookback INTEGER NOT NULL,
            n_expl_ep_per_it INTEGER NOT NULL,
            n_nonexpl_ep_per_it INTEGER NOT NULL,
            experiment_length_in_ep INTEGER NOT NULL,
            n_actor_pretraining INTEGER NOT NULL,
            n_critic_training_per_loop_iteration INTEGER NOT NULL,
            n_actor_training_per_loop_iteration INTEGER NOT NULL,
            CONSTRAINT UC_mainloop_config UNIQUE (
                restore_path,
                episode_length,
                lookback,
                n_expl_ep_per_it,
                n_nonexpl_ep_per_it,
                experiment_length_in_ep,
                n_actor_pretraining,
                n_critic_training_per_loop_iteration,
                n_actor_training_per_loop_iteration
            ),
            CONSTRAINT CC_divisible_n_sim CHECK (
                n_expl_ep_per_it % n_sim = 0 AND n_nonexpl_ep_per_it % n_sim = 0
            ),
            CONSTRAINT CC_divisible_n_ep CHECK (
                experiment_length_in_ep % (n_expl_ep_per_it + n_nonexpl_ep_per_it) = 0
            )
        );''',

    "hyperparameters_configs": '''
        CREATE TABLE IF NOT EXISTS hyperparameters_configs (
            hyperparameters_config_id INTEGER PRIMARY KEY AUTO_INCREMENT,
            hierarchization_config_id INTEGER NOT NULL,
            exploration_config_id INTEGER NOT NULL,
            discount_factor DECIMAL(12,7) NOT NULL,
            noise_magnitude_limit DECIMAL(12,7) NOT NULL,
            hierarchization_coef DECIMAL(12,7) NOT NULL,
            actor_learning_rate DECIMAL(12,7) NOT NULL,
            critic_learning_rate DECIMAL(12,7) NOT NULL,
            batch_size INTEGER NOT NULL,
            smoothing DECIMAL(12,7) NOT NULL,
            CONSTRAINT UC_mainloop_config UNIQUE (
                hierarchization_config_id,
                exploration_config_id,
                discount_factor,
                noise_magnitude_limit,
                hierarchization_coef,
                actor_learning_rate,
                critic_learning_rate,
                batch_size,
                smoothing
            ),
            FOREIGN KEY (hierarchization_config_id)
                    REFERENCES hierarchization_configs(hierarchization_config_id)
                    ON DELETE CASCADE,
            FOREIGN KEY (exploration_config_id)
                    REFERENCES exploration_configs(exploration_config_id)
                    ON DELETE CASCADE
        );''',

    "experiment_configs": '''
        CREATE TABLE IF NOT EXISTS experiment_configs (
            experiment_config_id INTEGER PRIMARY KEY AUTO_INCREMENT,
            mainloop_config_id INTEGER NOT NULL,
            hyperparameters_config_id INTEGER NOT NULL,
            repetitions_total INTEGER NOT NULL,
            repetitions_remaining INTEGER NOT NULL,
            repetitions_running INTEGER NOT NULL,
            repetitions_done INTEGER NOT NULL,

            FOREIGN KEY (mainloop_config_id)
                REFERENCES mainloop_configs(mainloop_config_id)
                ON DELETE CASCADE,
            FOREIGN KEY (hyperparameters_config_id)
                REFERENCES hyperparameters_configs(hyperparameters_config_id)
                ON DELETE CASCADE,
            CONSTRAINT UC_experiment_config UNIQUE (
                mainloop_config_id,
                hyperparameters_config_id
            )
        );''',

    "experiments": '''
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY AUTO_INCREMENT,
            experiment_config_id INTEGER NOT NULL,
            PRNGKey_start INTEGER NOT NULL,
            date_time_start DATETIME NOT NULL,
            date_time_stop DATETIME,
            hourly_pricing DECIMAL(12,7),
            path VARCHAR(255) NOT NULL,
            finished INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (experiment_config_id)
                REFERENCES experiment_configs(experiment_config_id)
                ON DELETE CASCADE,
            CONSTRAINT UC_PRNGKey UNIQUE (
                experiment_config_id,
                PRNGKey_start
            )
        );''',

    "results": '''
        CREATE TABLE IF NOT EXISTS results (
                experiment_id INTEGER NOT NULL,
                loop_iteration INTEGER NOT NULL,
                episode_nb INTEGER NOT NULL,
                training_episode_return DECIMAL(12,7),
                testing_episode_return DECIMAL(12,7),
                exploration_param DECIMAL(12,7) NOT NULL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE,
                CONSTRAINT UC_loop_iteration UNIQUE (
                    experiment_id,
                    loop_iteration
                ),
                CONSTRAINT UC_episode_nb UNIQUE (
                    experiment_id,
                    episode_nb
                )
        );''',

    "collections": '''
        CREATE TABLE IF NOT EXISTS collections (
            collection VARCHAR(255) PRIMARY KEY
        );''',

    "collections_experiment_config": '''
        CREATE TABLE IF NOT EXISTS collections_experiment_config (
            collection VARCHAR(255) NOT NULL,
            experiment_config_id INTEGER NOT NULL,
            FOREIGN KEY (collection)
                REFERENCES collections(collection),
            FOREIGN KEY (experiment_config_id)
                REFERENCES experiment_configs(experiment_config_id)
                ON DELETE CASCADE,
            CONSTRAINT UC_collection_exp_config_id UNIQUE (
                collection,
                experiment_config_id
            )
        );'''
}


class Database:
    def __init__(self, db_name, user='root', password='', host='127.0.0.1'):
        self._logger = logging.getLogger("Database")
        self._logger.info(f"[database] opening {host}")
        self.host = host
        self.conn = sql.connect(host=host, user=user, password=password)
        self.cursor = self.conn.cursor()

        ########################################################################
        try: # USE the database 'db_name'
            self.cursor.execute(f"USE {db_name}")
        except sql.Error as err:
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                self._logger.info(f"Database {db_name} does not exists. Creating it!")
                try: # to create the database
                    self.cursor.execute(f"CREATE DATABASE {db_name} DEFAULT CHARACTER SET 'utf8'")
                except mysql.connector.Error as err:
                    self._logger.critical(f"Failed creating database {db_name}: {err}")
                self._logger.info(f"Database {db_name} created successfully.")
                self.conn.database = db_name
            else:
                self._logger.critical(f"Unknown error when trying to 'USE {db_name}': {err}")
                raise err
        ########################################################################

        for table_name, command in TABLES.items():
            try:
                self.cursor.execute(command)
            except sql.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    self._logger.info(f"Table {table_name} already exists.")
                else:
                    self._logger.critical(err.msg)
                    raise err
            else:
                self._logger.info(f'Table {table_name} created')


    def add_to_collection(self, experiment_config_id, collection, protect=True):
        self.insert_into("collections", {
            "collection": collection
        }, protect=protect)
        self.insert_into("collections_experiment_config", {
            "collection": collection,
            "experiment_config_id": experiment_config_id
        }, protect=protect)

    def insert_mainloop_config(self,
            restore_path, n_sim, episode_length, lookback, n_expl_ep_per_it,
            n_nonexpl_ep_per_it, experiment_length_in_ep, n_actor_pretraining,
            n_critic_training_per_loop_iteration,
            n_actor_training_per_loop_iteration, protect=True):
        self.insert_into("mainloop_configs", {
            "restore_path": restore_path,
            "n_sim": n_sim,
            "episode_length": episode_length,
            "lookback": lookback,
            "n_expl_ep_per_it": n_expl_ep_per_it,
            "n_nonexpl_ep_per_it": n_nonexpl_ep_per_it,
            "experiment_length_in_ep": experiment_length_in_ep,
            "n_actor_pretraining": n_actor_pretraining,
            "n_critic_training_per_loop_iteration": n_critic_training_per_loop_iteration,
            "n_actor_training_per_loop_iteration": n_actor_training_per_loop_iteration,
        }, protect=protect)
        id = self.select_into("mainloop_configs", ["mainloop_config_id"], {
            "restore_path": restore_path,
            "episode_length": episode_length,
            "lookback": lookback,
            "n_expl_ep_per_it": n_expl_ep_per_it,
            "n_nonexpl_ep_per_it": n_nonexpl_ep_per_it,
            "experiment_length_in_ep": experiment_length_in_ep,
            "n_actor_pretraining": n_actor_pretraining,
            "n_critic_training_per_loop_iteration": n_critic_training_per_loop_iteration,
            "n_actor_training_per_loop_iteration": n_actor_training_per_loop_iteration,
        })[0][0]
        self._logger.info(f"The new entry has id {id}")
        return id

    def insert_hyperparameters_config(self,
            hierarchization_config_id, exploration_config_id, discount_factor,
            noise_magnitude_limit, hierarchization_coef, actor_learning_rate,
            critic_learning_rate, batch_size, smoothing, protect=True):
        self.insert_into("hyperparameters_configs", {
            "hierarchization_config_id": hierarchization_config_id,
            "exploration_config_id": exploration_config_id,
            "discount_factor": discount_factor,
            "noise_magnitude_limit": noise_magnitude_limit,
            "hierarchization_coef": hierarchization_coef,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "batch_size": batch_size,
            "smoothing": smoothing,
        }, protect=protect)
        id = self.select_into("hyperparameters_configs", ["hyperparameters_config_id"], {
            "hierarchization_config_id": hierarchization_config_id,
            "exploration_config_id": exploration_config_id,
            "discount_factor": discount_factor,
            "noise_magnitude_limit": noise_magnitude_limit,
            "hierarchization_coef": hierarchization_coef,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "batch_size": batch_size,
            "smoothing": smoothing,
        })[0][0]
        self._logger.info(f"The new entry has id {id}")
        return id

    def insert_experiment_config(self,
            mainloop_config_id, hyperparameters_config_id, repetitions_total,
            protect=True):
        self.insert_into("experiment_configs", {
            "mainloop_config_id": mainloop_config_id,
            "hyperparameters_config_id": hyperparameters_config_id,
            "repetitions_total": repetitions_total,
            "repetitions_remaining": repetitions_total,
            "repetitions_running": 0,
            "repetitions_done": 0,
        }, protect=protect)
        id = self.select_into("experiment_configs", ["experiment_config_id"], {
            "mainloop_config_id": mainloop_config_id,
            "hyperparameters_config_id": hyperparameters_config_id,
        })[0][0]
        self._logger.info(f"The new entry has id {id}")
        return id

    def insert_experiment(self, experiment_config_id, PRNGKey_start, date_time_start,
            hourly_pricing, path, protect=True):
        self.insert_into("experiments", {
            "experiment_config_id": experiment_config_id,
            "PRNGKey_start": PRNGKey_start,
            "date_time_start": date_time_start,
            "hourly_pricing": hourly_pricing,
            "path": path,
        }, protect=protect)
        id = self.select_into("experiments", ["experiment_id"], {
            "experiment_config_id": experiment_config_id,
            "PRNGKey_start": PRNGKey_start,
        })[0][0]
        self._logger.info(f"The new entry has id {id}")
        return id

    def insert_exploration_config(self, type, N, interpolation_type, upsilon_t0,
            upsilon_tN, exploration_prob_t0, exploration_prob_tN,
            softmax_temperature_t0, softmax_temperature_tN, protect=True):
        self.insert_into("exploration_configs", {
            "type": type,
            "N": N,
            "interpolation_type": interpolation_type,
            "upsilon_t0": upsilon_t0,
            "upsilon_tN": upsilon_tN,
            "exploration_prob_t0": exploration_prob_t0,
            "exploration_prob_tN": exploration_prob_tN,
            "softmax_temperature_t0": softmax_temperature_t0,
            "softmax_temperature_tN": softmax_temperature_tN,
        }, protect=protect)
        id = self.select_into("exploration_configs", ["exploration_config_id"], {
            "type": type,
            "N": N,
            "interpolation_type": interpolation_type,
            "upsilon_t0": upsilon_t0,
            "upsilon_tN": upsilon_tN,
            "exploration_prob_t0": exploration_prob_t0,
            "exploration_prob_tN": exploration_prob_tN,
            "softmax_temperature_t0": softmax_temperature_t0,
            "softmax_temperature_tN": softmax_temperature_tN,
        })[0][0]
        self._logger.info(f"The new entry has id {id}")
        return id

    def insert_hierarchization_config(self, n_actors, hierarchization_config, protect=True):
        self.insert_into("hierarchization_configs", {
            "n_actors": n_actors,
            "hierarchization_config": pickle.dumps(hierarchization_config),
        }, protect=protect)
        id = self.select_into("hierarchization_configs", ["hierarchization_config_id"], {
            "n_actors": n_actors,
            "hierarchization_config": pickle.dumps(hierarchization_config),
        })[0][0]
        self._logger.info(f"The new entry has id {id}")
        return id

    def insert_result(self, experiment_id, loop_iteration, episode_nb,
        training_episode_return, testing_episode_return, exploration_param, protect=True):
        self.insert_into("results", {
            "experiment_id": experiment_id,
            "loop_iteration": loop_iteration,
            "episode_nb": episode_nb,
            "training_episode_return": float(training_episode_return),
            "testing_episode_return": float(testing_episode_return),
            "exploration_param": float(exploration_param),
        }, protect=protect)

    def insert_into(self, table_name, name_values_dict, protect=True):
        duplicate = False
        self._logger.info(f'Inserting new data into table {table_name} ({protect=})')
        command = ""
        command += f"INSERT INTO {table_name}(\n    "
        command += ",\n    ".join(name_values_dict.keys())
        command += f"\n) VALUES ({','.join([' %s'] * len(name_values_dict))})"
        try:
            self.cursor.execute(command, tuple(name_values_dict.values()))
        except sql.errors.IntegrityError as e:
            if e.errno == errorcode.ER_DUP_ENTRY:
                duplicate = True
                if protect:
                    self._logger.critical(f"Trying to violate UNIQUE constraint ({table_name=})")
                    raise e
                else:
                    self._logger.info(f"The entry is already present in {table_name}")
            else:
                raise e
        self.conn.commit()
        return duplicate

    def select_into(self, table_name, columns, name_values_dict):
        tmp = []
        for name, val in name_values_dict.items():
            if val is None:
                tmp.append(f"{name} IS NULL")
            else:
                tmp.append(f"{name}=%s")

        command = f"SELECT {','.join(columns)} FROM {table_name} WHERE (\n    "
        command += ' AND\n    '.join(tmp)
        command += '\n)'
        self.cursor.execute(command, tuple(x for x in name_values_dict.values() if x is not None))
        return self.cursor.fetchall()

    def get_dataframe(self, command):
        return pd.read_sql(command, self.conn)

    def get_data(self, command):
        self.cursor.execute(command)
        return self.cursor.fetchall()

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
        command = f"UPDATE experiments SET finished=1,date_time_stop='{date_time_stop}',repetitions_done = repetitions_done + 1 WHERE experiment_id={experiment_id}"
        self.cursor.execute(command)
        self.conn.commit()

    def get_hierarchization_config(self, hierarchization_config_id):
        command = f'SELECT hierarchization_config FROM hierarchization_configs WHERE hierarchization_config_id={hierarchization_config_id} LIMIT 0, 1'
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
        command = f'SELECT * FROM exploration_configs WHERE exploration_config_id={exploration_config_id} LIMIT 0, 1'
        self.cursor.execute(command)
        (
            type,
            N,
            interpolation_type,
            upsilon_t0,
            upsilon_tN,
            exploration_prob_t0,
            exploration_prob_tN,
            softmax_temperature_t0,
            softmax_temperature_tN,
        ) = self.cursor.fetchone()[1:]
        return ExplorationConfig(type, N, interpolation_type, float(upsilon_t0), float(upsilon_tN),
            float(exploration_prob_t0), float(exploration_prob_tN), float(softmax_temperature_t0),
            float(softmax_temperature_tN))

    @contextmanager
    def get_a_job(self, path, hourly_pricing):
        command = f'SELECT * FROM experiment_configs WHERE repetitions_remaining > 0 LIMIT 0, 1'
        self.cursor.execute(command)
        res = self.cursor.fetchone() # tuple containing the data (one row)
        if res is None:
            yield None
            return
        (
            experiment_config_id,
            mainloop_config_id,
            hyperparameters_config_id,
            repetitions_total,
            repetitions_remaining,
            repetitions_running,
            repetitions_done,
        ) = res
        self._logger.info(f"get_a_job: {experiment_config_id=}")
        command = f'SELECT * FROM mainloop_configs WHERE mainloop_config_id=%s LIMIT 0, 1'
        self.cursor.execute(command, (mainloop_config_id,))
        res = self.cursor.fetchone()
        (
            mainloop_config_id,
            restore_path,
            n_sim,
            episode_length,
            lookback,
            n_expl_ep_per_it,
            n_nonexpl_ep_per_it,
            experiment_length_in_ep,
            n_actor_pretraining,
            n_critic_training_per_loop_iteration,
            n_actor_training_per_loop_iteration,
        ) = res
        command = f'SELECT * FROM hyperparameters_configs WHERE hyperparameters_config_id=%s LIMIT 0, 1'
        self.cursor.execute(command, (hyperparameters_config_id,))
        res = self.cursor.fetchone()
        (
            hyperparameters_config_id,
            hierarchization_config_id,
            exploration_config_id,
            discount_factor,
            noise_magnitude_limit,
            hierarchization_coef,
            actor_learning_rate,
            critic_learning_rate,
            batch_size,
            smoothing,
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
        experiment_id = self.insert_experiment(experiment_config_id, PRNGKey_start, datetime.now(), hourly_pricing, path)
        hierarchization_config = self.get_hierarchization_config(hierarchization_config_id)
        exploration_config = self.get_exploration_config(exploration_config_id)
        agent_args = (
            float(discount_factor),
            float(noise_magnitude_limit),
            hierarchization_config,
            float(hierarchization_coef),
            float(actor_learning_rate),
            float(critic_learning_rate),
            ACTION_DIM,
        )
        experiment_args = (n_sim, batch_size, float(smoothing), episode_length)
        mainloop_args = (
            PRNGKey_start,
            lookback,
            n_expl_ep_per_it,
            n_nonexpl_ep_per_it,
            experiment_length_in_ep,
            n_actor_pretraining,
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
        except:
            self._logger.critical(f"An exception has occured, deleting the experiment {experiment_id}")
            self.delete_experiment(experiment_id)
            self._logger.critical(f"An exception has occured, incrementing the 'remaining' counter ({experiment_config_id=})")
            command = f'''UPDATE experiment_configs
                          SET
                            repetitions_remaining = repetitions_remaining + 1
                          WHERE
                            experiment_config_id={experiment_config_id}
                          '''
            self.cursor.execute(command)
            self.conn.commit()
            raise
        finally:
            command = f'''UPDATE experiment_configs
                          SET
                            repetitions_running = repetitions_running - 1
                          WHERE
                            experiment_config_id={experiment_config_id}
                          '''
            self.cursor.execute(command)
            self.conn.commit()
        self.register_termination(experiment_id, datetime.now())


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    db = Database(db_name=sys.argv[1], user=sys.argv[2], password=sys.argv[3])

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
    n_actor_pretraining = 1000
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

    mainloop_config_id = db.insert_mainloop_config(
        restore_path,
        n_sim,
        episode_length,
        lookback,
        n_expl_ep_per_it,
        n_nonexpl_ep_per_it,
        experiment_length_in_ep,
        n_actor_pretraining,
        n_critic_training_per_loop_iteration,
        n_actor_training_per_loop_iteration,
    )

    hyperparameters_config_id = db.insert_hyperparameters_config(
        hierarchization_config_id,
        exploration_config_id,
        discount_factor,
        noise_magnitude_limit,
        hierarchization_coef,
        actor_learning_rate,
        critic_learning_rate,
        batch_size,
        smoothing,
    )

    experiment_config_id = db.insert_experiment_config(
        mainloop_config_id,
        hyperparameters_config_id,
        repetitions_total,
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
