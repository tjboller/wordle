import multiprocessing as mp
from functools import partial
from random import sample
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
import os
from utils import logger

WORDLE_WORD_LENGTH = 5
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSSIBLE_WORDS_FREQ = pd.read_csv(f'{PARENT_DIR}/data/five_letter_words_freq.csv')
POSSIBLE_WORDS = list(POSSIBLE_WORDS_FREQ['word'])

LOGGER = logger.get_logger(__name__)


def get_error_str_after_guess(answer, guess):
    error_str = ''
    for answer_letter, guess_letter in zip(answer,guess):
        if answer_letter == guess_letter:
            error_str += 'g'

        elif guess_letter in answer:
            error_str += 'y'

        elif guess_letter not in answer:
            error_str += 'r'

    return error_str


def get_words_after_guess(error_str, guess, possible_words=None):
    assert len(error_str) == 5, "invalid error str 1"
    assert set(error_str).difference(set(['g', 'r', 'y'])) == set([]), "invalid error str 2"
    words = POSSIBLE_WORDS if possible_words is None else possible_words
    for position, error_letter, guess_letter in zip(
            range(WORDLE_WORD_LENGTH), error_str, guess):

        if error_letter == 'g':
            words = [word for word in words
                     if word[position] == guess_letter]

        elif error_letter == 'y':
            words = [word for word in words
                     if ((guess_letter in word) and
                         (word[position] != guess_letter))]

        elif error_letter == 'r':
            words = [word for word in words
                     if guess_letter not in word]

    return words


def get_entropy_for_guess_and_correct_word(
        answer, guess,
        possible_words=None,
        proportional_answer_prob=False
    ):

    possible_words = (POSSIBLE_WORDS if possible_words is None
                      else possible_words)

    error_str = get_error_str_after_guess(answer=answer, guess=guess)
    remaining_words = get_words_after_guess(
        error_str=error_str, guess=guess, possible_words=possible_words)

    if proportional_answer_prob:
        possible_freq = POSSIBLE_WORDS_FREQ[
            POSSIBLE_WORDS_FREQ['word'].isin(remaining_words)
        ]
        return entropy(possible_freq['count'] / sum(possible_freq['count']))
    else:
        return len(remaining_words)


def get_distr_for_guess(
        guess, proportional_answer_prob=False,
        possible_words=None,
        sample_size=None, verbose=False
    ):

    distr = []
    prob = []

    possible_words = (POSSIBLE_WORDS if possible_words is None
                      else possible_words)
    possible_freq = POSSIBLE_WORDS_FREQ[
        POSSIBLE_WORDS_FREQ['word'].isin(possible_words)
    ]

    if sample_size is None:
        sample_words = possible_words
    else:
        weight = 'prob' if proportional_answer_prob else None
        if proportional_answer_prob:
            sample_words = possible_freq['word'].sample(
                sample_size, weights=weight)
        else:
            sample_words = sample(possible_words, sample_size)

    progress_bar = tqdm if verbose else blank_fn
    for answer in progress_bar(sample_words):

        if proportional_answer_prob:
            prob.append(possible_freq[possible_freq['word'] == answer].prob.values[0])
        else:
            prob.append(1/len(sample_words))

        distr.append(
            get_entropy_for_guess_and_correct_word(
                answer=answer,
                guess=guess,
                possible_words=possible_words,
                proportional_answer_prob=proportional_answer_prob))

    return distr, prob


def get_entropy_for_guess(
        guess, possible_words=None, proportional_answer_prob=False,
        sample_size=None, verbose=False,
    ):

    distr, prob = get_distr_for_guess(
        possible_words=possible_words,
        guess=guess,
        proportional_answer_prob=proportional_answer_prob,
        sample_size=sample_size,
        verbose=verbose
    )

    return sum(np.array(distr) * np.array(prob))


def get_all_distributions(
        possible_words, sample_size=None, proportional_answer_prob=None, verbose=True):

    # it's very fast under 1000 possible words
    verbose = False if not verbose or len(possible_words) < 1000 else verbose
    progress_bar = tqdm if verbose else blank_fn

    pool = mp.Pool(processes=mp.cpu_count()-1)
    f = partial(
        _distr_func,
        possible_words=possible_words,
        sample_size=sample_size,
        proportional_answer_prob=proportional_answer_prob
    )
    return dict(progress_bar(pool.imap_unordered(f, possible_words),
                     total=len(possible_words)))


def get_ranked_guesses(possible_words=None, sample_size=None, proportional_answer_prob=None):
    distrs = get_all_distributions(
        possible_words=possible_words,
        sample_size=sample_size,
        proportional_answer_prob=proportional_answer_prob
    )
    ranked_words = pd.Series({
        word: np.mean(distr)
        for word, distr in distrs.items()
    })

    ranked_words_df = pd.DataFrame(ranked_words, columns=['entropy'])
    ranked_words_df.index.name = 'word'
    ranked_words_df = ranked_words_df.reset_index()
    ranked_words_df = (
        pd.merge(POSSIBLE_WORDS_FREQ, ranked_words_df)
        .sort_values(['entropy', 'count'], ascending=[True, False])
    )
    return ranked_words_df.reset_index(drop=True)


def play_game(first_guess, answer=None, possible_words=None,
              proportional_answer_prob=None, verbose=True):

    possible_words = (POSSIBLE_WORDS if possible_words is None
                          else possible_words)
    guess = first_guess
    turns = 0
    while len(possible_words) != 1:

        if answer is not None:
            error_str = get_error_str_after_guess(answer=answer, guess=guess)
        else:
            print(
                'enter the error string. g for green, y for yellow, a r otherwise. No spaces')
            error_str = input()

        possible_words = get_words_after_guess(
            error_str=error_str, guess=guess, possible_words=possible_words)

        num_words = len(possible_words)
        ranked_new_guesses = get_ranked_guesses(
            possible_words=possible_words,
            proportional_answer_prob=proportional_answer_prob
        )

        if verbose:
            LOGGER.info(f'Guess: {guess}, {num_words} words remain')
            LOGGER.info(f'Top Next Guesses: \n{ranked_new_guesses[:10]}\n')

        guess = ranked_new_guesses.iloc[0]['word']
        turns += 1
    return turns


def _distr_func(guess, sample_size, proportional_answer_prob, possible_words):
    entropy, _ = get_distr_for_guess(
        possible_words=possible_words,
        guess=guess,
        proportional_answer_prob=proportional_answer_prob,
        sample_size=sample_size
    )
    return guess, entropy


def _entropy_func(guess, possible_words=None, proportional_answer_prob=False,
        sample_size=None):

    entropy = get_entropy_for_guess(
        guess=guess,
        possible_words=possible_words,
        sample_size=sample_size,
        proportional_answer_prob=proportional_answer_prob
    )
    return guess, entropy


def get_entropy_for_guesses(
        guesses, possible_words=None, proportional_answer_prob=False,
        sample_size=None, verbose=True):

    possible_words = (POSSIBLE_WORDS if possible_words is None
                      else possible_words)
    progress_bar = tqdm if verbose else blank_fn

    pool = mp.Pool(processes=mp.cpu_count() - 1)
    f = partial(
        _entropy_func,
        possible_words=possible_words,
        sample_size=sample_size,
        proportional_answer_prob=proportional_answer_prob
    )
    return dict(progress_bar(pool.imap_unordered(f, guesses),
                             total=len(guesses)))

def blank_fn(x, *args, **kwargs):
    return x