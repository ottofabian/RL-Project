import logging


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
