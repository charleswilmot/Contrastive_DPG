def default_case(table_name):
    primary_key = table_name.rstrip("s") + "_id"
    def default(db, ids):
        in_str = f'({",".join(ids)})'
        print(db.get_dataframe(f'''
        SELECT * FROM {table_name} WHERE {primary_key} IN {in_str}
        '''))
    return default


def hierarchization_configs(db, ids):
    for id in ids:
        id = int(id)
        hc = db.get_hierarchization_config(id)
        for level, (size, dmin, dmax, slopemin, slopemax) in enumerate(hc):
            print(f'{id=: 2d}   {level=}   {size=: 2d}   {dmin=:.3f}   {dmax=:.3f}   {slopemin=:.3f}   {slopemax=:.3f}')
        print("\n\n")


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
    parser.add_argument('--user', default='ubuntu', help='username for MySQL DB')
    parser.add_argument('--password', default='aqwsedcft', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DPG_v2', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')
    parser.add_argument('table', help='table name')
    parser.add_argument('ids', nargs='+', type=str, help='id to query')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)


    table_to_func = {
        "experiment_configs": default_case("experiment_configs"),
        "hyperparameters_configs": default_case("hyperparameters_configs"),
        "exploration_configs": default_case("exploration_configs"),
        "mainloop_configs": default_case("mainloop_configs"),
        "hierarchization_configs": hierarchization_configs,
    }

    table_to_func[args.table](db, args.ids)
