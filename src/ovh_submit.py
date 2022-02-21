from ovh import *

if __name__ == '__main__':
    import argparse
    import logging
    import time

    logger = logging.getLogger("ovh_submit")


    parser = argparse.ArgumentParser(description='Creates N workers on the OVH cloud.')
    parser.add_argument('--db-name', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--no-master', action='store_true', help='Number of workers to create')
    parser.add_argument('N', help='Number of workers to create', type=int)
    parser.add_argument('db_path', help='path to a DB file to upload on master')

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
        upload_db(ssh_clients["master"], args.db_name, args.db_path)
