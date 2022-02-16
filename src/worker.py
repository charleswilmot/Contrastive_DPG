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


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    db = Database(sys.argv[1])


    done = False
    while not done:


        log_path = sys.argv[2]
        ids = [int(match.group(1)) for x in os.listdir(log_path) if (match := re.match('([0-9]+)_[a-zA-Z]+[0-9]+_[0-9]+-[0-9]+', x))]
        if ids:
            exp_id = 1 + max(ids)
        else:
            exp_id = 0
        run_name = f'{exp_id:03d}_{datetime.datetime.now():%b%d_%H-%M}'
        path = f'{log_path}/{run_name}'
        os.makedirs(path, exist_ok=True)

        file_handler = logging.FileHandler(f"{path}/output.log")
        formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


        with db.get_a_job(path) as args:
            if args is not None:
                agent = Agent(*args.agent)
                with Experiment(*args.experiment, agent) as experiment:
                    experiment.mainloop(*args.mainloop)
            else:
                done = True

        logger.removeHandler(file_handler)
