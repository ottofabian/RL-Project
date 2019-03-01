import logging
import sys

import gym

from Experiments.util.ColorLogger import enable_color_logging
from Experiments.util.logger_util import show_cmd_args
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO
import time
import numpy as np

from PILCO.util.util import parse_args, evaluate_policy


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

    # load the models if "args.weight_dir" is given
    if args.weight_dir:
        # make sure that the dir ends with an "/"
        if args.weight_dir[-1] != '/':
            args.weight_dir += '/'

        # load the policy
        pilco.load_policy(f"{args.weight_dir}policy.p")
        if not args.test:
            # load the remaining models and stats to continue training
            pilco.load_dynamics(f"{args.weight_dir}dynamics.p")
            pilco.load_data(f"{args.weight_dir}state-action.npy", f"{args.weight_dir}state-delta.npy")

    if args.test:
        evaluate_policy(pilco.policy, pilco.env, max_action=np.array([5]))

    pilco.run()


if __name__ == '__main__':
    main()
