"""Runs basic analysis of data and provides visualisation"""

# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import pandas_profiling  # pylint: disable=required import for profile module

from streamlit_pandas_profiling import st_profile_report
from app.components.models import cart_model
from app.components.utils import widget_functions

# MAIN SCRIPT --------------------------------------------------------------
@st.cache(suppress_st_warning=True)
def data_summary(file):
    st.header('Data Summary')
    st.write(
        'Analysis of uploaded data using '
        '[streamlit pandas profiling](https://github.com/okld/streamlit-pandas-profiling) [MIT License]'
    )

    label_info = widget_functions.create_select_label_widget()

    model = cart_model.RegressionTree()
    model.get_data(file)
    model.get_input_options()
    model.create_label_feature(label_feature=label_info['selected_label'], censoring=label_info['censor_state'], duration=label_info['duration'])
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

    # Read subset options from uploaded data template
    subset_dict = model.get_subset_options_dict()

    # Widget to interactively update subset options
    if label_info['selected_label'] == 'Failure within given duration [y/n]':
        survival_state = st.selectbox('Create subset based on graft function', ['working', 'failed', 'both'])
        # No need to create subset if 'both' selected'
        if survival_state == 'working':
            subset_dict[model.label_feature]=[1]
        elif survival_state == 'failed':
            subset_dict[model.label_feature]=[0]

    # create updated dataframe based on subset options
    model.create_dataframe_subset(subset_dict)

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
