"""Runs basic analysis of data and provides visualisation"""

# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import pandas_profiling  # pylint: disable=required import for profile module

from streamlit_pandas_profiling import st_profile_report
from app.components.models import cart_model

# MAIN SCRIPT --------------------------------------------------------------
@st.cache(suppress_st_warning=True)
def data_summary(file):
    st.header('Data Summary')
    st.write(
        'Analysis of uploaded data using '
        '[streamlit pandas profiling](https://github.com/okld/streamlit-pandas-profiling) [MIT License]'
    )

    selected_label = 'Survival time [days]'
    censor_state = st.checkbox('Select to activate left censoring')
    if censor_state is True:
        duration = st.number_input('Censor duration [days]', min_value=1, max_value=365, format='%i', value=365,
            help='This input will be used for left censoring of survival duration')
    else:
        duration = 1000

    model = cart_model.RegressionTree()
    model.get_data(file)
    model.get_input_options()
    model.create_label_feature(label_feature=selected_label, censoring=censor_state, duration=duration)
    available_variable_list = model.available_input_features + [model.label_feature]
    selected_variables = st.multiselect(
        'Features to analyze (select at least one):',
        available_variable_list,
        default=available_variable_list,
        help='run pandas profiling on the following features extracted from the uploaded file'
        )
    if len(selected_variables) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    df_to_analyze = model.dataframe[selected_variables]

    if st.session_state['train_state'] is False:
        st.session_state['continue_state'] = st.button('Run analysis')
    if st.session_state['continue_state'] is False:
        st.stop()

    while st.session_state['train_state'] is False:
        report = df_to_analyze.profile_report(
            html={
                'style':{'full_width':True}
                }
            )
        st_profile_report(report)

        st.session_state['train_state'] = True

    run_again = st.button('Run again')
    if run_again:
        st.session_state['train_state'] = False
