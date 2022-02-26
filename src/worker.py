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
    parser.add_argument('--db-name', default='Contrastive_DPG_v2', help='name for MySQL DB')
    parser.add_argument('--hourly-pricing', default=0.0889, help='hourly pricing of the OVH instance')

    args = parser.parse_args()

    db = Database(db_name=args.db_name, user=args.user, password=args.password, host=args.host)
    os.makedirs("../experiments/", exist_ok=True)


    done = False
    while not done:


        with db.get_a_job(args.hourly_pricing) as (job_path, job_args):
            if job_args is not None:

                os.makedirs(job_path, exist_ok=True)

                file_handler = logging.FileHandler(f"{job_path}/output.log")
                sys.stderr = open(f"{job_path}/error.log", "a")
                formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

                agent = Agent(*job_args.agent)
                with Experiment(*job_args.experiment, agent) as experiment:
                    experiment.mainloop(*job_args.mainloop)
            else:
                done = True

        logger.removeHandler(file_handler)
