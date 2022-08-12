"""Runs basic analysis of data and provides visualisation"""

# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import pandas_profiling  # pylint: disable=required import for profile module

from streamlit_pandas_profiling import st_profile_report

# MAIN SCRIPT --------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def process_options(model, selected_features, selected_label, censoring, duration):
    model.get_input_features(selected_features)
    model.create_label_feature(label_feature=selected_label, censoring=censoring, duration=duration)
    return model

def data_summary(file):
    st.header('Data Summary')
    st.write('Analysis of uploaded data using [streamlit pandas profiling](https://github.com/okld/streamlit-pandas-profiling) [MIT License]')

    df = file['Data']
    pr = df.profile_report()
    st_profile_report(pr)
