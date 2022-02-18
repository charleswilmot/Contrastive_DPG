from ovh import *

if __name__ == '__main__':
    import argparse
    import logging
    import time

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger = logging.getLogger("ovh_submit")


    parser = argparse.ArgumentParser(description='Creates N workers on the OVH cloud.')
    parser.add_argument('--db-name', default='Contrastive_DPG_debug', help='name for MySQL DB')
    parser.add_argument('--no-master', action='store_true', help='Number of workers to create')
    parser.add_argument('N', help='Number of workers to create', type=int)
    parser.add_argument('--set-names', nargs='+', help='names of the sets of experiments to add to the DB')

    args = parser.parse_args()

    novac = get_nova_client()

    logger.info(f"Creating {args.N} instances {'(no master)' if args.no_master else ''}")
    names = get_missing_names(novac, args.N, args.no_master)
    instances = create_instances(novac, names, wait=120)

    logger.info(f"Openning SSH connections")
    ssh_clients = get_ssh_clients(instances)

    for name, ssh_client in ssh_clients.items():
        logger.info(f"Waiting on {name}")
        wait_instance_ready(ssh_client)

    if not args.no_master:
        logger.info(f"Populating master's DB")
        populate_db(ssh_clients["master"], args.db_name, args.set_names)

    host = get_master_instance(novac).addresses["Ext-Net"][0]["addr"]
    for name, ssh_client in ssh_clients.items():
        logger.info(f"Starting {name}")
        start_worker(ssh_client, host, args.db_name)
        time.sleep(2)
