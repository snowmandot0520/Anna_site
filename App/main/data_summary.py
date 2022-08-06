# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from App.main.models.cart_model import RegressionTree
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# MAIN SCRIPT --------------------------------------------------------------
def data_summary(file, verbose):

    @st.cache(allow_output_mutation=True)
    def process_options(model, selected_features, selected_label, censoring):
        model.get_input_features(selected_features)
        model.get_label_feature(label_feature=selected_label, censoring=censoring)
        return model

    st.header('Data Summary')

    st.write('#### Basic survival analysis of uploaded data')
    selected_label = 'Survival time [days]'
    censor_state = st.checkbox('Select to activate left censoring to 1 year')

    # Create RegressionTree class and process options
    model = RegressionTree()
    model.get_data(file)

    available_input_options = model.get_input_options()  # return value is equal given same df for all subclass of BaseModel
    selected_features = st.multiselect('Input features (select at least one):', available_input_options)
    model = process_options(model, selected_features, selected_label, censor_state)

    if len(model.input_features) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    # Preprocess data
    st.write('### Dataset')
    model.process_input_options(verbose=False)

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

    # Univariate Models
    st.write('### Univariate Models')

    fig_uni, (ax1_uni, ax2_uni) = plt.subplots(1, 2, figsize=(12, 4))

    kmf = KaplanMeierFitter()
    kmf.fit(durations=np.asarray(model.y), event_observed=np.asarray(model.x['Event_observed']), label='Kaplan Meier Estimate')
    kmf.plot(ax=ax1_uni)
    ax1_uni.set_title('Kaplan Meier survival estimate')
    ax1_uni.set_xlabel('Time [days]')
    ax1_uni.set_ylabel('Survival probability')

    naf = NelsonAalenFitter()
    naf.fit(durations=np.asarray(model.y), event_observed=np.asarray(model.x['Event_observed']), label='Nelson Aalen Estimate')
    naf.plot_cumulative_hazard(ax=ax2_uni)
    ax2_uni.set_title('Nelson Aalen hazard estimate')
    ax2_uni.set_xlabel('Time [days]')
    ax2_uni.set_ylabel('Cumulative hazard')

    st.pyplot(fig_uni)