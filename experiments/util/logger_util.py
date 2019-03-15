import logging
import os
import sys
import datetime
from time import time


def enable_logging(logging_lvl=logging.DEBUG, save_log=False, logfile_prefix=""):
    """
    Enables Color logging on multi-platforms as well as in environments like jupyter notebooks

    :param logging_lvl: Given debug level for setting what messages to show. (logging.DEBUG is lowest)
    :param save_log: If true a log file will be created under ./logs
    :param logfile_prefix; Prefix for defining the name of the log file
    :return:
    """

    root = logging.getLogger()
    root.setLevel(logging_lvl)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging_lvl)

    # FORMAT from https://github.com/xolox/python-coloredlogs
    FORMAT = '%(asctime)s %(name)s[%(process)d] \033[1m%(levelname)s\033[0m %(message)s'

    # FORMAT="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    formatter = logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S")

    ch.setFormatter(formatter)
    root.addHandler(ch)

    if save_log:
        # include current timestamp in dataset export file
        timestmp = datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d-%H-%M-%S")
        formatter = logging.Formatter("%(asctime)s %(message)s")

        if not os.path.isdir("./experiments/logs/"):
            print(os.getcwd())
            os.mkdir("./experiments/logs/")

        file_handler = logging.FileHandler("./experiments/logs/" + logfile_prefix + timestmp + ".log", mode='a')
        file_handler.setFormatter(formatter)
        # avoid spamming the log file, only log INFO , WARNING, ERROR events
        file_handler.setLevel(logging.INFO)

        root.addHandler(file_handler)


def show_cmd_args(args):
    """
    Logs all given cmd-line parameters to logging.info() given the args handle.
    :param args: Cmd-line arguments
    :return:
    """
    # log all given hyper-parameter command line settings
    logging.info("Command line parameters:")
    for arg in vars(args):
        # for cmd line parameters - is used as a separator in actual python _ is used
        cmd_parameter = arg.replace("_", "-")
        logging.info(f"--{cmd_parameter} {getattr(args, arg)}")
