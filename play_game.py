import argparse
import os

from utils import model
from utils import logger

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


LOGGER = logger.get_logger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--suppress', dest='suppress', action='store_true',
        help='suppress printing to options the screen'
    )
    parser.add_argument(
        '-i', '--initial_guess',
        help='the first word to guess')

    parser.add_argument(
        '-a', '--answer',
        help='the hidden answer')

    parser.add_argument(
        '-p', '--proportional_answer',
        help='whether you want to assume proportional likelihood of answers to frequency of use')

    parser.set_defaults(
        suppress=False,
        initial_guess='lares',
        answer=None,
        proportional_answer=False
    )
    args = parser.parse_args()
    verbose = not args.suppress
    num_tries = model.play_game(
        first_guess=args.initial_guess,
        answer=args.answer,
        proportional_answer_prob=args.proportional_answer,
        verbose=verbose
    )
    if args.answer:
        print(f'{args.answer}, {num_tries}')
    else:
        print(num_tries)

