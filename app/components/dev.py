# pylint: skip-file
# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
from app.components.models import beta_gp

# MAIN SCRIPT --------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def process_options(model, selected_features, selected_label, censoring, duration):
    model.get_input_features(selected_features)
    model.create_label_feature(label_feature=selected_label, censoring=censoring, duration=duration)
    return model

def dev(file, verbose):
    st.write('#### Development: [Beta] Gaussian Process')
    # Basic label select option
    selected_label = st.selectbox(
        'Prediction target:',
        ['Survival time [days]', 'Failure within given duration [y/n]']
        )
    if selected_label == 'Failure within given duration [y/n]':
        duration = st.number_input('Failed within [days]', min_value=1, max_value=365, format='%i', value=365,
            help='This input will be used to create boolean labels for classification models')
        censor_state = False
    elif selected_label == 'Survival time [days]':
        censor_state = st.checkbox('Left censoring', value=True)
        if censor_state is True:
            duration = st.number_input('Censor duration [days]', min_value=1, max_value=365, format='%i', value=365,
                help='This input will be used for left censoring of labels for regression models')
        else:
            duration = 1000

    model = beta_gp.GaussianProcess()
    model.verbose = verbose
    model.get_data(file)
    available_input_options = model.get_input_options()  # return value is equal given same df for all subclass of BaseModel
    selected_features = st.multiselect('Input features (select at least one):', available_input_options)
    model = process_options(model, selected_features, selected_label, censor_state, duration)

    if len(model.input_features) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    # Preprocess data
    st.write("### Dataset")
    model.process_input_options(verbose=False)

    st.write("### Train / Test split")
    test_proportion = st.slider('Test set proprotion?', min_value=0.1, max_value=0.5, step=0.1)
    model.train_test_split(test_proportion=test_proportion)

    # TODO implement gaussian process model
