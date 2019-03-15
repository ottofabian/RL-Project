import logging
import sys
import time
import quanser_robots

from a3c.a3c import A3C
from a3c.util.util import parse_args
from experiments.util.logger_util import enable_logging

from experiments.util.logger_util import show_cmd_args


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])
    enable_logging(logging_lvl=logging.DEBUG, save_log=not args.no_log,
                   logfile_prefix="A3C_" + args.env_name + "_")

    logging.info(
        f'Start experiment for {args.env_name} at {time.strftime("%m/%d/%Y, %Hh:%Mm:%Ss", time.gmtime(time.time()))}')

    show_cmd_args(args)

    a3c = A3C(args)
    a3c.run()
