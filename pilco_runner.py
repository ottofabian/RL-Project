import logging
import sys

import gym

from experiments.util.ColorLogger import enable_color_logging
from experiments.util.logger_util import show_cmd_args
from pilco.cost_function.saturated_loss import SaturatedLoss
from pilco.pilco import PILCO
import time

from pilco.util.util import parse_args, evaluate_policy, load_model


def main():
    args = parse_args(sys.argv[1:])

    enable_color_logging(logging_lvl=logging.DEBUG, save_log=not args.no_log,
                         logfile_prefix="PILCO_" + args.env_name + "_")

    logging.info(
        f'Start experiment for {args.env_name} at {time.strftime("%m/%d/%Y, %Hh:%Mm:%Ss", time.gmtime(time.time()))}')

    # show given cmd-parameters
    show_cmd_args(args)
    env = gym.make(args.env_name)

    # make sure that the dir ends with an "/"
    if args.weight_dir:
        if args.weight_dir[-1] != '/':
            args.weight_dir += '/'

    if args.test:
        policy = load_model(f"{args.weight_dir}policy.p")
        evaluate_policy(policy, env, max_action=args.max_action, no_render=args.no_render)

    else:
        state_dim = env.observation_space.shape[0]
        loss = SaturatedLoss(state_dim=state_dim, target_state=args.target_state, weights=args.weights)
        pilco = PILCO(args, loss=loss)

        # load the models if "args.weight_dir" is given
        if args.weight_dir:
            pilco.load(args.weight_dir)

        pilco.run()


if __name__ == '__main__':
    main()
