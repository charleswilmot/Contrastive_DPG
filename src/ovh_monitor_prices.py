
if __name__ == '__main__':
    from database import Database
    import argparse
    import logging
    import sys
    import pandas as pd

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Populate database with experiments.')
    parser.add_argument('--user', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DPG_v2', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    df = db.get_dataframe('''
        SELECT
            experiment_id,
            date_time_start,
            date_time_stop,
            hourly_pricing
        FROM
            experiments
    ''')
    price = df.hourly_pricing * (df.date_time_stop - df.date_time_start) / pd.Timedelta('1 hour')

    results = pd.concat([df.experiment_id, price], axis=1, keys=["experiment_id", "price"])
    pd.set_option('display.max_rows', None)
    print(results)

    print("total:", results.price.sum())
