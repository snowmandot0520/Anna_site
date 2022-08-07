"""Runs basic analysis of data and provides visualisation"""

# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from app.components.models import cart_model
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# MAIN SCRIPT --------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def process_options(model, selected_features, selected_label, censoring, duration):
    model.get_input_features(selected_features)
    model.create_label_feature(label_feature=selected_label, censoring=censoring, duration=duration)
    return model

def data_summary(file):
    st.header('Data Summary')
    st.write('#### Basic survival analysis of uploaded data')
    selected_label = 'Survival time [days]'
    censor_state = st.checkbox('Select to activate left censoring')
    if censor_state is True:
        duration = st.number_input('Censor duration [days]', min_value=1, max_value=365, format='%i', value=365,
            help='This input will be used for left censoring of labels for regression models')
    else:
        duration = 1000

    model = cart_model.RegressionTree()
    model.get_data(file)
    available_input_options = model.get_input_options()
    selected_features = st.multiselect(
        'Input features (select at least one):',
        available_input_options
        )
    model = process_options(model, selected_features, selected_label, censoring=censor_state, duration=duration)

    if len(model.input_features) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    st.write('### Dataset')
    model.process_input_options()

    model.x['Event_observed'] = np.zeros(model.y.size)
    for idx, duration in model.y.iteritems():
        if duration < 365:
            model.x.at[idx, 'Event_observed'] = 1

    st.write('**Summary generated using pandas**')
    st.dataframe(model.dataframe.describe())

    show_data = st.checkbox('Select to show uploaded dataset')
    if show_data:
        st.write('**Uploaded dataset**')
        st.dataframe(model.dataframe)

    st.write('### Univariate Models')

    fig_uni, (ax1_uni, ax2_uni) = plt.subplots(1, 2, figsize=(12, 4))

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=np.asarray(model.y),
        event_observed=np.asarray(model.x['Event_observed']),
        label='Kaplan Meier Estimate'
        )
    kmf.plot(ax=ax1_uni)
    ax1_uni.set_title('Kaplan Meier survival estimate')
    ax1_uni.set_xlabel('Time [days]')
    ax1_uni.set_ylabel('Survival probability')

    naf = NelsonAalenFitter()
    naf.fit(
        durations=np.asarray(model.y),
        event_observed=np.asarray(model.x['Event_observed']),
        label='Nelson Aalen Estimate'
        )
    naf.plot_cumulative_hazard(ax=ax2_uni)
    ax2_uni.set_title('Nelson Aalen hazard estimate')
    ax2_uni.set_xlabel('Time [days]')
    ax2_uni.set_ylabel('Cumulative hazard')

    st.pyplot(fig_uni)
    