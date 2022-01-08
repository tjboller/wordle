import streamlit as st
from stqdm import stqdm
from utils import model
import pandas as pd
from io import BytesIO

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
st.set_page_config(layout='wide')


def validate_word(word):
    if word not in model.POSSIBLE_WORDS:
        st.error(f'{word} not a valid word')


def validate_error(error_str):
    if len(error_str) != 5 or set(error_str).difference(set('gry')) != set([]):
        st.error(f'{error_str} not a valid error str')



###########################
# Define Constants and functions
###########################

sample_size=500
st.title('Wordle Explorer')
st.set_option('deprecation.showPyplotGlobalUse', False)

###
# Wordle Helper
###
st.subheader('Solver')

st.text('Write multiple guesses with comma seperation')
st.text('For the errors - Use g for green, y for yellow, and r otherwise. For example "gyyyr, ryggy"')

st.session_state.possible_words = model.POSSIBLE_WORDS
col1, col2 = st.columns(2)

if "guesses_submitted" not in st.session_state:
    st.session_state.guesses_submitted = False

with col1:
    guesses = st.text_input('Write your guesses:')
with col2:
    error_strs = st.text_input('Write your errors:')

if st.button(label='Execute!', key=0) or st.session_state.guesses_submitted:
    st.session_state.guesses_submitted = True

if guesses and error_strs and st.session_state.guesses_submitted:
    guesses = guesses.replace(' ', '').split(',')
    error_strs = error_strs.replace(' ', '').split(',')
    possible_words = model.POSSIBLE_WORDS
    for guess, error in zip(guesses, error_strs):
        validate_error(error)
        validate_word(guess)
        possible_words = model.get_words_after_guess(
            error_str=error,
            guess=guess,
            possible_words=possible_words
        )
        num_words = len(possible_words)
        ranked_new_guesses = model.get_ranked_guesses(
            possible_words=possible_words
        )
        ranked_new_guesses.columns = [
            'Word', 'Word Use Freq', 'prob', 'Average Words Remaining After Guess'
        ]
        st.text(f'{num_words} possible words remaining')
        st.dataframe(ranked_new_guesses.drop(columns=['prob']))

###########################
# Start
###########################
st.subheader('Evaluation of Start Words')

###
# Best and worst
###
col1, col2 = st.columns(2)
with col1:
    st.text('Best Start Words')
    df_top = pd.read_csv('data/results/top_distributions.csv')
    df_top.columns = [
                'Word', 'Average Words Remaining After Guess'
            ]
    st.dataframe(df_top)

with col2:
    st.text('Worst Start Words')
    df_bottom = pd.read_csv('data/results/bottom_distributions.csv')
    df_bottom.columns = [
                'Word', 'Average Words Remaining After Guess'
            ]
    st.dataframe(df_bottom)


sample_size = st.select_slider(
    'Hidden Answer Sample Size',
    options=[10, 100, 200, 500, 1000, 5000, len(model.POSSIBLE_WORDS)],
    value=500)
start_words = st.multiselect('Start Guesses:', model.POSSIBLE_WORDS)


if "submitted" not in st.session_state:
    st.session_state.submitted = False

if st.button(label='Execute!') or st.session_state.submitted:
    data = {
        start_word:
            model.get_distr_for_guess(
                start_word,
                sample_size=sample_size
            )[0]
        for start_word in stqdm(start_words)
    }
    data = pd.DataFrame(
        pd.DataFrame.from_dict(data, orient='index').transpose(),
        columns=start_words)

    st.session_state.submitted = True

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.set_ylabel('Number of Answers')
    ax.set_xlabel('Density')
    sns.histplot(data=data, element='step', ax=ax, alpha=.3)

    summary_avg = pd.DataFrame(data.mean())
    summary_avg.columns = ['Average number of possible words']
    st.dataframe(summary_avg.style.format("{:,.1f}"))

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)




