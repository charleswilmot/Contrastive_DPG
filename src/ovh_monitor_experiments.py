
if __name__ == '__main__':
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
    parser.add_argument('--user', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    print(db.get_dataframe('''
        SELECT
            experiment_config_id,
            repetitions_total,
            repetitions_remaining,
            repetitions_running,
            repetitions_done
        FROM
            experiment_configs
    '''))
