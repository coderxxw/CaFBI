import sys
import warnings

from model import Error
from model.commands import create_argument_parser
from model.configuration import load_configuration
from model.logging import get_logger, enable_verbose_logging


def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        if arguments.verbose:
            enable_verbose_logging()
        if arguments.command == 'train':
            from model.commands.train import entrypoint
        elif arguments.command == 'eval':
            from model.commands.eval import entrypoint
        elif arguments.command == 'case':
            from model.commands.case import entrypoint
        else:
            raise Error(
                f'model does not support command {arguments.command}')
        params = load_configuration(arguments.configuration)
        params['best_epoch'] = arguments.best_epoch
        entrypoint(params)
    except Error as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
