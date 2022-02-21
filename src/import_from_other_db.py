if __name__ == '__main__':
    from database import Database
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

    parser.add_argument('--user1', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password1', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name1', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--host1', default='127.0.0.1', help='IP for MySQL DB')

    parser.add_argument('--user2', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password2', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name2', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--host2', default='127.0.0.1', help='IP for MySQL DB')

    args = parser.parse_args()

    db1 = Database(db_name=args.db_name1, user=args.user1, password=args.password1, host=args.host1)
    db2 = Database(db_name=args.db_name2, user=args.user2, password=args.password2, host=args.host2)

    db1.import_experiment_configs(db2)
