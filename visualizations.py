import streamlit as st
from stqdm import stqdm
from utils import model
import pandas as pd
from io import BytesIO

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
st.set_page_config(layout='wide')


###########################
# Define Constants and functions
###########################

sample_size=500
st.title('Wordle Explorer')
st.set_option('deprecation.showPyplotGlobalUse', False)

###
# Best and worst
###
col1, col2 = st.columns(2)
with col1:
    st.text('Best Start Words')
    st.dataframe(pd.read_csv('data/results/top_distributions.csv'))

with col2:
    st.text('Worst Start Words')
    st.dataframe(pd.read_csv('data/results/bottom_distributions.csv'))

###########################
# Start
###########################
st.subheader('Evaluation of Start Word')

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




