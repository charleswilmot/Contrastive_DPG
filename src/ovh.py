import os
import re
import time
import sys
import subprocess
import paramiko
from scp import SCPClient
from paramiko.client import SSHClient
from keystoneauth1.identity import v3
from keystoneauth1 import session
import novaclient.client
import novaclient.exceptions
import requests
# suppress warning
requests.packages.urllib3.disable_warnings()
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger = logging.getLogger("ovh")


def get_nova_client():
    auth = v3.Password(
        auth_url=os.environ['OS_AUTH_URL'],
        username=os.environ['OS_USERNAME'],
        password=os.environ['OS_PASSWORD'],
        user_domain_name=os.environ['OS_USER_DOMAIN_NAME'],
        project_domain_name=os.environ['OS_PROJECT_DOMAIN_NAME'],
    )
    sess = session.Session(auth=auth, verify=False)
    return novaclient.client.Client(
        2,
        session=sess,
        region_name=os.environ['OS_REGION_NAME'],
    )

def assert_no_instances_running(novac):
    if len(novac.servers.list()):
        raise RuntimeError("There are instances running")


def assert_instance_running(novac, instance_name):
    instances = novac.servers.list()
    names = [s.name for s in instances]
    if instance_name not in names:
        raise RuntimeError(f"Could not find the '{instance_name}' instance")


def assert_master_running(novac):
    assert_instance_running(novac, 'master')


def get_missing_names(novac, n, no_master):
    instances = novac.servers.list()
    names = [s.name for s in instances]
    ret = []
    if not no_master:
        if 'master' in names:
            raise RuntimeError("'master' instance already running, try with --no-master")
        else:
            ret.append('master')
            n -= 1
    found = 0
    i = 0
    while found < n:
        name = f'worker{i}'
        if name not in names:
            found += 1
            ret.append(name)
        i += 1
    return ret

def get_instance_by_name(novac, instance_name):
    assert_instance_running(novac, instance_name)
    instances = novac.servers.list()
    for instance in instances:
        if instance.name == instance_name:
            return instance

def get_master_instance(novac):
    return get_instance_by_name(novac, 'master')


def get_master_instance(novac):
    return get_instance_bay_name(novac, 'master')


def create_instances(novac, names, image_name="Ubuntu 20.04", flavor_name="c2-7", wait=120):
    image = novac.glance.find_image(image_name)
    flavor = novac.flavors.find(name=flavor_name)
    net = novac.neutron.find_network(name="Ext-Net")
    keypair = novac.keypairs.find(name="SSHKEY")
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


def get_ssh_clients(instances):
    return {
        name: get_ssh_client(instance)
        for name, instance in instances.items()
    }


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
        cd Code/Contrastive_DPG/src/
        python3 worker.py --user ubuntu --password aqwsedcft --db-name {db_name} --host {host} > /dev/null 2>&1 &
        '''
    )


def is_worker_running(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command('pgrep -f worker.py')
    exit_status = stdout.channel.recv_exit_status()
    return exit_status == 0


def get_uptime(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command('uptime -p')
    return stdout.readline().rstrip()


def scp_progress(filename, size, sent):
    logger.info(f'Downloading {filename}: received {int(100 * sent / size): 3d}% -- {sent: 9d}/{size: 9d}')


def download_db(ssh_master, db_name, db_path):
    stdin, stdout, stderr = ssh_master.exec_command(f'''
        mysqldump --user=ubuntu --password=aqwsedcft {db_name} > /tmp/{db_name}.sql
    ''')
    for line in stdout.readlines():
        logger.info(f"download_db (stdout):    {line.rstrip()}")
    for line in stderr.readlines():
        logger.info(f"download_db (stderr):    {line.rstrip()}")
    scp = SCPClient(ssh_master.get_transport(), progress=scp_progress)
    os.makedirs(db_path, exist_ok=True)
    ids = tuple(
        int(m.group(1))
        for filename in os.listdir(db_path)
        if (m := re.match(f'{db_name}.sql.([0-9]+)', filename))
    )
    count = (max(ids) + 1) if ids else 0
    filesuffix = f'.{count}'
    filename = f'{db_name}.sql{filesuffix}'
    file_path = f'{db_path}/{filename}'
    scp.get(f"/tmp/{db_name}.sql", file_path)


def rsync_experiments(host,
        remote_experiments_path='/home/ubuntu/Code/Contrastive_DPG/experiments/',
        local_experiments_path='../experiments/remote/'):
    with subprocess.Popen([
            "rsync",
            "-avz",
            f"ubuntu@{host}:{remote_experiments_path.rstrip('/')}/",
            f"{local_experiments_path.rstrip('/')}",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for line in process.stdout:
            logger.info(f'rsync_experiments (stdout):    {line.rstrip()}')
        for line in process.stderr:
            logger.info(f'rsync_experiments (stderr):    {line.rstrip()}')
