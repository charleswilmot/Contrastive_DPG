if __name__ == '__main__':
    import argparse
    import logging
    from ovh import *

    logger = logging.getLogger("ovh/restart_workers")

    parser = argparse.ArgumentParser(description='Restart workers.')
    parser.add_argument('--db-name', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--host', default='127.0.0.1', help='IP for MySQL DB')
    parser.add_argument('instances', nargs='+', help='list of instances names to restart the workers on')

    args = parser.parse_args()

    novac = get_nova_client()


    for instance_name in args.instances:
        assert_instance_running(novac, instance_name)

    for instance_name in args.instances:
        logger.info(f"restarting worker on {instance_name}")
        instance = get_instance_by_name(novac, instance_name)
        ssh_client = get_ssh_client(instance)
        start_worker(ssh_client, args.host, args.db_name)
