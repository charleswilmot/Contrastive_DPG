import os
import time
import logging
import sys
import paramiko
from paramiko.client import SSHClient
from keystoneauth1.identity import v3
from keystoneauth1 import session
import novaclient.client
import novaclient.exceptions
import requests
# suppress warning
requests.packages.urllib3.disable_warnings()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger = logging.getLogger("ovh_submit")


def assert_no_instances_running(novac):
    if len(novac.servers.list()):
        raise RuntimeError("There are instances running")


def create_instances(novac, n, image_name="Ubuntu 20.04", flavor_name="c2-7", wait=120):
    assert_no_instances_running(novac)
    if n < 1:
        raise RuntimeError("Cannot start {n} instances")
    image = novac.glance.find_image(image_name)
    flavor = novac.flavors.find(name=flavor_name)
    net = novac.neutron.find_network(name="Ext-Net")
    keypair = novac.keypairs.find(name="SSHKEY")
    names = ["master"] + [f"worker{i}" for i in range(n - 1)]
    instances = {}

    for name in names:
        with open("../local/cloud_config.yaml", 'r') as userdata:
            instance = novac.servers.create(
                name,
                image,
                flavor,
                key_name=keypair.name,
                nics=[{'net-id': net.id}],
                userdata=userdata
            )
            instances[name] = instance
    logger.info(f'sleeping {wait} secs')
    time.sleep(wait)
    return instances


def get_ssh_client(instance):
    client = SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    while 'Ext-Net' not in instance.addresses:
        logger.info(f"instance {instance.name} not ready for SSH, sleeping 120 sec")
        time.sleep(120)
    host = instance.addresses["Ext-Net"][0]["addr"]
    login = f'ubuntu@{host}'
    logger.info(f"connection via SSH to {login}")
    client.connect(hostname=host, username='ubuntu')
    return client


def wait_instance_ready(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command(
        "tail -f /var/log/cloud-init-output.log | sed '/Installation complete/ q'"
    )
    for line in stderr.readlines():
        logger.info(f"wait_instance_ready (stderr):    {line.rstrip()}")


def populate_db(ssh_client, db_name, experiment_sets):
    stdin, stdout, stderr = ssh_client.exec_command(
        f'''
        cd Code/Contrastive_DPG/src/;
        python3 register_experiments.py --user ubuntu --password aqwsedcft --db-name {db_name} {" ".join(experiment_sets)}
        '''
    )
    for line in stdout.readlines():
        logger.info(f"populate_db (stdout):    {line.rstrip()}")
    for line in stderr.readlines():
        logger.info(f"populate_db (stderr):    {line.rstrip()}")


def start_worker(ssh_client, host, db_name):
    stdin, stdout, stderr = ssh_client.exec_command(
        f'''
        export COPPELIASIM_ROOT=/home/ubuntu/Softwares/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/
        export LD_LIBRARY_PATH=/home/ubuntu/Softwares/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/
        export QT_QPA_PLATFORM_PLUGIN_PATH=/home/ubuntu/Softwares/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/
        export COPPELIASIM_MODEL_PATH=/home/ubuntu/Code/Contrastive_DPG/3d_models/
        cd Code/Contrastive_DPG/src/;
        python3 worker.py --user ubuntu --password aqwsedcft --db-name {db_name} --host {host}
        ''',
        environment={
            "COPPELIASIM_ROOT": "/home/ubuntu/Softwares/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/",
            "LD_LIBRARY_PATH": "/home/ubuntu/Softwares/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/home/ubuntu/Softwares/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/",
            "COPPELIASIM_MODEL_PATH": "/home/ubuntu/Code/Contrastive_DPG/3d_models/",
        }
    )
    # for line in stdout.readlines():
    #     logger.info(f"start_worker (stdout):    {line.rstrip()}")
    # for line in stderr.readlines():
    #     logger.info(f"start_worker (stderr):    {line.rstrip()}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Creates N workers on the OVH cloud.')
    parser.add_argument('--db-name', default='Contrastive_DQN_debug', help='name for MySQL DB')
    parser.add_argument('N', help='Number of workers to create', type=int)
    parser.add_argument('set_names', nargs='+', help='names of the sets of experiments to add to the DB')

    args = parser.parse_args()

    auth = v3.Password(
        auth_url=os.environ['OS_AUTH_URL'],
        username=os.environ['OS_USERNAME'],
        password=os.environ['OS_PASSWORD'],
        user_domain_name=os.environ['OS_USER_DOMAIN_NAME'],
        project_domain_name=os.environ['OS_PROJECT_DOMAIN_NAME'],
    )
    sess = session.Session(auth=auth, verify=False)
    novac = novaclient.client.Client(
        2,
        session=sess,
        region_name=os.environ['OS_REGION_NAME'],
    )

    logger.info(f"Creating {args.N} instances")
    instances = create_instances(novac, args.N, wait=120)

    logger.info(f"Openning SSH connections")
    ssh_clients = {
        name: get_ssh_client(instance)
        for name, instance in instances.items()
    }

    for name, ssh_client in ssh_clients.items():
        logger.info(f"Waiting on {name}")
        wait_instance_ready(ssh_client)

    logger.info(f"Populating master's DB")
    populate_db(ssh_clients["master"], args.db_name, args.set_names)

    host = instances["master"].addresses["Ext-Net"][0]["addr"]
    for name, ssh_client in ssh_clients.items():
        logger.info(f"Starting {name}")
        start_worker(ssh_client, host, args.db_name)
