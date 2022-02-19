
if __name__ == '__main__':
    from database import Database
    import argparse
    import logging
    import sys
    import pandas as pd
    from collections import OrderedDict

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Monitor collections.')
    parser.add_argument('--user', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DPG_debug', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)

    collections = [x[0] for x in db.get_data('''
        SELECT DISTINCT
            collection
        FROM
            collections
    ''')]

    for collection in collections:
        exp_config_infos = db.get_dataframe(f'''
            SELECT e.experiment_config_id, h.*, m.* FROM experiment_configs AS e
                INNER JOIN hyperparameters_configs AS h
                ON e.hyperparameters_config_id = h.hyperparameters_config_id
                INNER JOIN mainloop_configs AS m ON e.mainloop_config_id = m.mainloop_config_id
                WHERE
                    e.experiment_config_id IN (
                        SELECT
                            experiment_config_id
                        FROM
                            collections_experiment_config
                        WHERE
                            collection='{collection}'
                    )
        ''')
        variation_columns = exp_config_infos.loc[:, (exp_config_infos != exp_config_infos.iloc[0]).any()]
        scores = pd.DataFrame()
        for index, info in variation_columns.iterrows():
            returns = db.get_dataframe(f'''
                SELECT
                    episode_nb,
                    AVG(training_episode_return),
                    AVG(testing_episode_return)
                FROM
                    results
                WHERE
                    experiment_id IN (
                        SELECT
                            experiment_id
                        FROM
                            experiments
                        WHERE
                            experiment_config_id={info.experiment_config_id} AND
                            finished=1
                    )
                GROUP BY
                    episode_nb
            ''')
            if len(returns):
                score_total = returns.drop('episode_nb', 1).mean()
                score_begining = returns.drop('episode_nb', 1)[returns.episode_nb < 4000].mean()
                score_end = returns.drop('episode_nb', 1)[returns.episode_nb > 12000].mean()
                row = pd.Series(OrderedDict(
                    # collection=collection,
                    test=score_total["AVG(testing_episode_return)"],
                    train=score_total["AVG(training_episode_return)"],
                    test_begining=score_begining["AVG(testing_episode_return)"],
                    train_begining=score_begining["AVG(training_episode_return)"],
                    test_end=score_end["AVG(testing_episode_return)"],
                    train_end=score_end["AVG(training_episode_return)"],
                ))
                row = info[2:].append(row)
                scores = scores.append(row, ignore_index=True, sort=False)
        scores = scores.reindex(row.index, axis=1)
        print("###", collection, '###')
        print(scores)
        print("\n\n")
