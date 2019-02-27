import logging
import sys

import gym

from Experiments.util.ColorLogger import enable_color_logging
from Experiments.util.logger_util import show_cmd_args
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO
import time

from PILCO.util.util import parse_args


def main():

    args = parse_args(sys.argv[1:])

    enable_color_logging(logging_lvl=logging.DEBUG, save_log=args.save_log,
                         logfile_prefix="PILCO_" + args.env_name + "_")

    logging.info(
        f'Start experiment for {args.env_name} at {time.strftime("%m/%d/%Y, %Hh:%Mm:%Ss", time.gmtime(time.time()))}')

    # show given cmd-parameters
    show_cmd_args(args)

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]

    loss = SaturatedLoss(state_dim=state_dim, target_state=args.target_state, W=args.weights)
    pilco = PILCO(args, loss=loss)

    # load the models if they are given
    if args.dynamics_path:
        pilco.load_dynamics(args.dynamics_path)
    if args.policy_path:
        pilco.load_policy(args.policy_path)

    pilco.run()


if __name__ == '__main__':
    main()
