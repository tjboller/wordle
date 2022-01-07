import argparse
import pickle
import os
from tqdm import tqdm
from utils import model
import numpy as np
import pandas as pd

import multiprocessing as mp
from functools import partial

from utils import logger

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


LOGGER = logger.get_logger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose',
        type=int,
        help='whether to print out the game'
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
        verbose=True,
        initial_guess='tares',
        answer=None,
        proportional_answer=False
    )
    args = parser.parse_args()

    if args.answer is None:
        pass
    else:
        model.play_game(
            first_guess=args.initial_guess,
            answer=args.answer,
            proportional_answer_prob=args.proportional_answer,
            verbose=args.verbose
        )






