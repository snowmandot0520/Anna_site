# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st

from App.main.models.support_vector_machine_model import SupportVectorClassifier, SupportVectorRegression
from App.main.models.random_forest_model import RandomForest
from App.main.models.cart_model import RegressionTree
from App.main.models.cox_ph_model import CoxProportionalHazardsRegression


# MAIN SCRIPT --------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def process_options(model, selected_features, selected_label, censoring, rfe):
    model.get_input_features(selected_features)
    model.get_label_feature(label_feature=selected_label, censoring=censoring)
    model.rfe = rfe
    return model

def interactive_run(file, verbose):

    with st.sidebar:
        st.write('\n')
        show_data = st.checkbox('Show raw data')

        selected_label = st.sidebar.selectbox('Prediction target:', ['Survival time [days]', 'Failure within 1 year [y/n]'])
        censor_state = st.checkbox('Left censoring to 1 year', value=True)

        # Classification task
        if selected_label == 'Failure within 1 year [y/n]':
            selected_model = st.selectbox('Classification model:', ['Support Vector Machine', 'Random Forest'])

            if selected_model == 'Support Vector Machine':
                model = SupportVectorClassifier()
                model.class_weight = st.selectbox('Class Weight:', ['Balanced', None],
                                                  help='Balanced weight helps in datasets with disproportionate sample size between classes')
                model.c_param = st.number_input('C param?', min_value=0.01, max_value=100.0, value=1.0)
            elif selected_model == 'Random Forest':
                model = RandomForest()
                model.n_estimators = st.number_input('Number of trees?', min_value=10, max_value=1000)
                model.max_depth = st.number_input('Depth of trees?', min_value=1, max_value=10)
                model.criterion = st.selectbox('Criterion?', ['gini', 'entropy'])
                model.class_weight = 'balanced_subsample'

        # Regression task
        elif selected_label == 'Survival time [days]':
            selected_model = st.selectbox('Regression model:', ['Cox Proportional Hazards', 'Support Vector Machine', 'Regression Tree'])

            if selected_model == 'Cox Proportional Hazards':
                model = CoxProportionalHazardsRegression()

            elif selected_model == 'Support Vector Machine':
                model = SupportVectorRegression()
                model.c_param = st.number_input('C param?', min_value=0.01, max_value=100.0, value=1.0)

            elif selected_model == 'Regression Tree':
                model = RegressionTree()

        model.get_data(file)
        available_input_options = model.get_input_options()

        selected_features = st.sidebar.multiselect('Input features (select at least one):',available_input_options)
        if selected_model == 'Cox Proportional Hazards':
            feature_elimination = False
        else:
            feature_elimination = st.checkbox('Recursive feature elimination?', value=True, help='Must select at least 2 features')

        model = process_options(model, selected_features, selected_label, censoring=censor_state, rfe=feature_elimination)

    if show_data:
        st.subheader('Show raw data')
        st.dataframe(model.dataframe)

    if len(model.input_features) == 0:
        st.sidebar.warning('❗Please select at least one input feature to run analysis.')
        st.stop()
    elif len(model.input_features) == 1 and feature_elimination == True:
        st.sidebar.warning('❗Please select at least two input feature to recursive feature elimination')
        st.stop()

    st.write('#### Dataset size')
    model.process_input_options(verbose=verbose)
    if selected_model == 'Cox Proportional Hazards':
        model.create_event_status(duration_days=365)

    st.write('#### Train / Test split')
    model.train_test_split(test_proportion=st.slider('Test set proprotion?', min_value=0.1, max_value=0.5, step=0.1, value=0.3))

    if st.session_state['train_state'] == False:
        st.session_state['continue_state'] = st.button('Run model')

    if st.session_state['continue_state'] == False:
        st.stop()

    while st.session_state['train_state'] == False:

        model.build_estimator()
        if selected_model == 'Cox Proportional Hazards':
            model.combine_outcome_data()
        st.write('#### Train model')
        with st.spinner('training in progress...'):
            model.train()
        st.success('Done!')
        st.session_state['train_state'] = True

    st.write('#### Model performance on test set')
    model.evaluate(verbose=verbose)
    model.save_log()

    st.write('#### Model output visualisation')
    if selected_model == 'Cox Proportional Hazards':
        selected_covariates = st.multiselect('Plot partial effects on outcome for following:', model.x.columns.drop('Event_observed'))
        model.visualize(selected_covariates)
    else:
        model.visualize()
    model.save_fig()

    st.write('#### Save options')
    # create download option from browser
    model.create_log_download_button()
    model.create_fig_download_button()

    train_again = st.button('Train again')
    if train_again:
        st.session_state['train_state'] = False