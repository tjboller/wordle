import argparse
import pickle
import os
from utils import model
import numpy as np
import pandas as pd


from utils import logger

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE_PATH = f"{CURR_DIR}/data/results/sample_distributions.p"
TOP_FILE_PATH = f"{CURR_DIR}/data/results/top_distributions.csv"
BOTTOM_FILE_PATH = f"{CURR_DIR}/data/results/bottom_distributions.csv"

LOGGER = logger.get_logger(__name__)


def dict_to_csv(dict_, path):
    x = pd.Series(dict_).sort_values()
    x.index.name = 'word'
    x.name = 'count'
    x.to_csv(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--sample_size',
        type=int,
        help='whether you want to assume proportional likelihood of answers to frequency of use'
    )

    parser.add_argument(
        '-p', '--proportional_answer',
        help='whether you want to assume proportional likelihood of answers to frequency of use')

    parser.add_argument(
        '-c', '--use_cache',
        help='whether you want to use the cached data')

    parser.add_argument(
        '-t', '--num_top_samples',
        help='how many of the top sampled words to check completely')

    parser.add_argument(
        '-b', '--num_bottom_samples',
        help='how many of the bottom sampled words to check completely')

    parser.set_defaults(
        sample_size=100,
        proportional_answer=False,
        use_cache=True,
        num_top_samples=250,
        num_botton_samples=10
    )
    args = parser.parse_args()

    if args.use_cache:
        LOGGER.info('getting already ran sample distributions...')
        sample_results = pickle.load(open(SAMPLE_FILE_PATH, 'rb'))
    else:
        LOGGER.info('getting sample distributions...')
        sample_results = model.get_all_distributions(
            possible_words=model.POSSIBLE_WORDS_FREQ,
            sample_size=args.sample_size,
            proportional_answer_prob=args.proportional_answer
        )
        pickle.dump(
            sample_results,
            open(SAMPLE_FILE_PATH, "wb")
        )

    sample_means = pd.Series({
        word: np.mean(distr)
        for word, distr in sample_results.items()
    }).sort_values()

    LOGGER.info(f'getting full results for top {args.num_top_samples} samples')
    top_inputs = sample_means[:args.num_top_samples].index
    dict_to_csv(model.get_entropy_for_guesses(
        guesses=top_inputs,
        proportional_answer_prob=args.proportional_answer
    ), TOP_FILE_PATH)

    LOGGER.info(f'getting full results for bottom {args.num_botton_samples} samples')
    bottom_inputs = sample_means[-args.num_botton_samples:].index
    dict_to_csv(model.get_entropy_for_guesses(
        guesses=bottom_inputs,
        proportional_answer_prob=args.proportional_answer
    ), BOTTOM_FILE_PATH)



