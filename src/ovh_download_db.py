if __name__ == '__main__':
    from ovh import *
    import argparse
    import logging

    logger = logging.getLogger("ovh/download_db")

    parser = argparse.ArgumentParser(description='Creates N workers on the OVH cloud.')
    parser.add_argument('--db-name', default='Contrastive_DPG', help='name for MySQL DB')
    parser.add_argument('--db-path', default='../databases/', help='Path to database backup files')

    args = parser.parse_args()

    novac = get_nova_client()
    master = get_master_instance(novac)
    ssh_master = get_ssh_client(master)
    download_db(ssh_master, args.db_name, args.db_path)

    for instance in novac.servers.list():
        logger.info(f"Downloading experiments from {instance.name}")
        rsync_experiments(
            instance.addresses["Ext-Net"][0]["addr"],
            local_experiments_path=f'../experiments/remote/{instance.name}'
        )
