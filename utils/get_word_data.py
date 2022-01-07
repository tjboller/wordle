
import pandas as pd
import pickle
import os

WORDLE_WORD_LENGTH = 5
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":

    # valid scrabble words from
    # https://drive.google.com/file/d/1oGDf1wjWp5RF_X9C7HoedhIWMh5uJs8s/view
    with open(f'{PARENT_DIR}/data/scrabble_words.txt', 'r') as words_f:
        words = words_f.read().splitlines()

    # word frequencies from from Peter Norvig's compilation of
    # the 1/3 million most frequent English words.
    # https://norvig.com/ngrams/count_1w.txt
    word_freq = pd.read_csv(
        f'{PARENT_DIR}/data/word_freq.txt',
        sep='\t', names=['word', 'count'])

    # save all five letter words in a text file for ease of use
    valid_words = [word.lower() for word in words if len(word) == 5]
    with open(f'{PARENT_DIR}/data/five_letter_words.txt', 'w') as f:
        f.write('\n'.join(valid_words))

    valid_words_df = pd.DataFrame(valid_words, columns=['word'])
    five_letter_freq = pd.merge(valid_words_df, word_freq, on='word', how='left')
    five_letter_freq = five_letter_freq.fillna({'count': word_freq['count'].min()/2})
    five_letter_freq['prob'] = five_letter_freq['count']/sum(five_letter_freq['count'])
    five_letter_freq['word'] = [
        word.lower() for word in five_letter_freq['word']
    ]
    five_letter_freq.to_csv(
        f'{PARENT_DIR}/data/five_letter_words_freq.csv', index=False)
