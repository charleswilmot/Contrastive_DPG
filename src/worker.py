import matplotlib
matplotlib.use('TkAgg') # necessary to avoid conflict with Coppelia's Qt
from database import Database


def Agent(*args):
    print(args)


class Experiment:
    def __init__(self, *args):
        print(args)
    def mainloop(self, *args):
        print(args)


if __name__ == '__main__':
    db = Database('/tmp/debug.db')

    path = '/tmp/nothing'
    with db.get_a_job(path) as args:
        if args is not None:
            agent = Agent(*args.agent)
            experiment = Experiment(*args.experiment, agent)
            experiment.mainloop(*args.mainloop)
