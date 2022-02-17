import matplotlib
matplotlib.use('Agg') # necessary to avoid conflict with Coppelia's Qt
from database import Database
import logging
import sys
import os
import re
import datetime
from experiment import Experiment
from agent import Agent
import argparse


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Executes jobs from the DB until non remain.')
    parser.add_argument('--host', default='127.0.0.1', help='MySQL DB IP address')
    parser.add_argument('--user', default='root', help='username for MySQL DB')
    parser.add_argument('--password', default='', help='password for MySQL DB')
    parser.add_argument('--db-name', default='Contrastive_DQN_debug', help='name for MySQL DB')
    parser.add_argument('--log-path', default='../experiments/', help='path where experiments are logged')
    parser.add_argument('--hourly-pricing', default=0.0889, help='hourly pricing of the OVH instance')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)


    done = False
    while not done:


        log_path = args.log_path
        os.makedirs(log_path, exist_ok=True)
        ids = [int(match.group(1)) for x in os.listdir(log_path) if (match := re.match('([0-9]+)_[a-zA-Z]+[0-9]+_[0-9]+-[0-9]+', x))]
        if ids:
            exp_id = 1 + max(ids)
        else:
            exp_id = 0
        run_name = f'{exp_id:03d}_{datetime.datetime.now():%b%d_%H-%M}'
        path = f'{log_path}/{run_name}'
        os.makedirs(path, exist_ok=True)

        file_handler = logging.FileHandler(f"{path}/output.log")
        sys.stderr = open(f"{path}/error.log", "a")
        formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


        with db.get_a_job(path, args.hourly_pricing) as args:
            if args is not None:
                agent = Agent(*args.agent)
                with Experiment(*args.experiment, agent) as experiment:
                    experiment.mainloop(*args.mainloop)
            else:
                done = True

        logger.removeHandler(file_handler)
