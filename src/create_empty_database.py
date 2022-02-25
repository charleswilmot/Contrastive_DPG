
if __name__ == '__main__':
    from database import Database
    import argparse
    import logging
    import sys
    import subprocess
    import re
    import os

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

    args = parser.parse_args()

    # check if db exists
    with subprocess.Popen([
            "mysqlshow",
            f"--user={args.user}",
            f"--password={args.password}",
            args.db_name
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        exists = False
        for line in process.stdout:
            if line.startswith(f"Database: {args.db_name}"):
                exists = True
                logger.critical("Database already exists!")
    # if not, create it
    with subprocess.Popen([
            "mysql",
            f"--user={args.user}",
            f"--password={args.password}",
            f'-e "create database {args.db_name}"',
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for line in process.stdout:
            logger.info(f'creating db (stdout):    {line.rstrip()}')
        for line in process.stderr:
            logger.info(f'creating db (stderr):    {line.rstrip()}')
    # create tables
    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    # dump to a new db file

    ids = tuple(
        int(m.group(1))
        for filename in os.listdir("../databases/")
        if (m := re.match(f'{args.db_name}.sql.([0-9]+)', filename))
    )
    count = (max(ids) + 1) if ids else 0

    with open(f"../databases/{args.db_name}.sql.{count}", 'wb') as f:
        with subprocess.Popen([
                "mysqldump",
                f"--user={args.user}",
                f"--password={args.password}",
                args.db_name,
            ], stdout=f, stderr=subprocess.PIPE) as process:
            for line in process.stderr:
                logger.info(f'dumping db (stderr):    {line.rstrip()}')
