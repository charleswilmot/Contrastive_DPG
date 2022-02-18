
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from database import Database
    import argparse
    import logging
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Populate database with experiments.')
    parser.add_argument('--user', default='root', help='username for MySQL DB')
    parser.add_argument('--password', default='', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DPG_debug', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')
    parser.add_argument('experiment_config_id', type=int, help='id of the experiment config to plot')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    df = db.get_dataframe(f'''
        SELECT
            episode_nb,
            AVG(training_episode_return),
            AVG(testing_episode_return),
            STD(training_episode_return),
            STD(testing_episode_return)
        FROM
            results
        WHERE
            experiment_id IN (
                SELECT
                    experiment_id
                FROM
                    experiments
                WHERE
                    experiment_config_id={args.experiment_config_id}
            )
        GROUP BY
            episode_nb
    ''')

    x = df["episode_nb"].values
    y_test = df["AVG(testing_episode_return)"].values
    y_train = df["AVG(training_episode_return)"].values

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_test, label='test')
    ax.plot(x, y_train, label='train')
    ax.set_xlabel("#episodes")
    ax.set_ylabel("episode return")
    ax.legend()
    plt.show()
