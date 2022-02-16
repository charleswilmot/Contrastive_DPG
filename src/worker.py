import matplotlib
matplotlib.use('Agg') # necessary to avoid conflict with Coppelia's Qt
from database import Database


def Agent(*args):
    print(args)


class Experiment:
    def __init__(self, *args):
        print(args)
    def mainloop(self, *args):
        print(args)


if __name__ == '__main__':
    import sys
    import os
    import datetime

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

        with db.get_a_job(path) as args:
            if args is not None:
                agent = Agent(*args.agent)
                with Experiment(*args.experiment, agent) as experiment:
                    experiment.mainloop(*args.mainloop)
            else:
                done = True
