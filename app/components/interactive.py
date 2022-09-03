"""Run models in an interactive session with runtime training
and evaluation based on parameters selected.
"""

# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st

from app.components.models import support_vector_machine_model
from app.components.models import random_forest_model
from app.components.models import cart_model
from app.components.models import cox_ph_model
from app.components.models import multi_layer_perceptron_model
from app.components.utils import widget_functions

# MAIN SCRIPT --------------------------------------------------------------
def interactive(file, verbose):
    with st.sidebar:
        st.write('\n')
        show_data = st.checkbox('Show raw data')

        label_info = widget_functions.create_select_label_widget()

        # Classification task
        if label_info['selected_label'] == 'Failure within given duration [y/n]':
            selected_model = st.selectbox(
                'Classification model:',
                ['Support Vector Machine', 'Random Forest', 'Multi-layer Perceptron']
                )
            if selected_model == 'Support Vector Machine':
                model = support_vector_machine_model.SupportVectorClassifier()
                model.class_weight = st.selectbox(
                    'Class Weight:',
                    ['Balanced', None],
                    help='Balanced weight helps in datasets with disproportionate \
                        sample size between classes'
                    )
                model.c_param = st.number_input(
                    'C param?',
                    min_value=0.01,
                    max_value=100.0,
                    value=1.0
                    )
            elif selected_model == 'Random Forest':
                model = random_forest_model.RandomForest()
                model.n_estimators = st.number_input(
                    'Number of trees?',
                    min_value=10,
                    max_value=1000
                    )
                model.max_depth = st.number_input('Depth of trees?', min_value=1, max_value=10)
                model.criterion = st.selectbox('Criterion?', ['gini', 'entropy'])
                model.class_weight = 'balanced_subsample'
            elif selected_model == 'Multi-layer Perceptron':
                model = multi_layer_perceptron_model.MultiLayerPerceptronClassifier()
                hidden_layer_dimensions=[]
                number_of_layers = st.number_input(
                    'Hidden layer depth?',
                    min_value=1, max_value=3, value=1,
                    help='Maximum of 3 hidden layers given computational resource limit')
                for hidden_layer in range(0, number_of_layers):
                    layer_size = st.number_input(f'size of layer {hidden_layer}?', min_value=1, max_value=100, value=10)
                    hidden_layer_dimensions.append(layer_size)
                model.hidden_layer_sizes = tuple(hidden_layer_dimensions)
                model.activation = st.selectbox('Non linear activation?', options=model.activation_list)
                model.alpha = st.select_slider('Alpha?', options=model.alpha_list, value=0.001)

        # Regression task
        elif label_info['selected_label'] == 'Survival time [days]':
            selected_model = st.selectbox(
                'Regression model:',
                ['Cox Proportional Hazards', 'Support Vector Machine', 'Regression Tree', 'Multi-layer Perceptron']
                )
            if selected_model == 'Cox Proportional Hazards':
                model = cox_ph_model.CoxProportionalHazardsRegression()
            elif selected_model == 'Support Vector Machine':
                model = support_vector_machine_model.SupportVectorRegression()
                model.c_param = st.number_input(
                    'C param?',
                    min_value=0.01,
                    max_value=100.0,
                    value=1.0
                    )
            elif selected_model == 'Regression Tree':
                model = cart_model.RegressionTree()
            elif selected_model == 'Multi-layer Perceptron':
                model = multi_layer_perceptron_model.MultiLayerPerceptronRegressor()
                hidden_layer_dimensions=[]
                number_of_layers = st.number_input(
                    'Hidden layer depth?',
                    min_value=1, max_value=3, value=1,
                    help='Maximum of 3 hidden layers given computational resource limit')
                for hidden_layer in range(0, number_of_layers):
                    layer_size = st.number_input(f'size of layer {hidden_layer}?', min_value=1, max_value=100, value=10)
                    hidden_layer_dimensions.append(layer_size)
                model.hidden_layer_sizes = tuple(hidden_layer_dimensions)
                model.activation = st.selectbox('Non linear activation?', options=model.activation_list)
                model.alpha = st.select_slider('Alpha?', options=model.alpha_list, value=0.001)

        model.verbose = verbose
        model.get_data(file)
        available_input_options = model.get_input_options()
        selected_features = st.sidebar.multiselect(
            'Input features (select at least one):',
            available_input_options
            )
        if selected_model in ['Cox Proportional Hazards', 'Multi-layer Perceptron']:
            feature_elimination = False
        else:
            feature_elimination = st.checkbox(
                'Recursive feature elimination?',
                value=True,
                help='Must select at least 2 features'
                )
        model = widget_functions.process_options(
            model=model,
            selected_features=selected_features,
            selected_label=label_info['selected_label'],
            censoring=label_info['censor_state'],
            duration=label_info['duration'],
            rfe=feature_elimination
            )

    subset_dict = {
        'Transplant type': ['DBD kidney transplant', 'DCD kidney transplant', 'LRD kidney transplant', 'LUD kidney transplant']
        }
    model.create_dataframe_subset(subset_dict)

    if show_data:
        st.subheader('Show raw data')
        st.dataframe(model.dataframe)

    if len(model.input_features) == 0:
        st.sidebar.warning('❗Please select at least one input feature to run analysis.')
        st.stop()
    elif len(model.input_features) == 1 and feature_elimination is True:
        st.sidebar.warning('❗Please select at least two input feature \
            to recursive feature elimination')
        st.stop()

    model.process_input_options()

    if label_info['selected_label'] == 'Survival time [days]':
        st.write('### Univariate survival curve')
        model.create_event_status(duration_days=365)
        model.plot_univariate_survival_curve()
        model.remove_event_status()

    st.write('#### Train / Test split')
    model.train_test_split(
        test_proportion=st.slider('Test set proprotion?', min_value=0.1, max_value=0.5, step=0.1, value=0.3)
        )

    if st.session_state['train_state'] is False:
        st.session_state['continue_state'] = st.button('Run model')
    if st.session_state['continue_state'] is False:
        st.stop()

    while st.session_state['train_state'] is False:
        model.build_estimator()
        if selected_model == 'Cox Proportional Hazards':
            model.combine_outcome_data()
        st.write('#### Train model')
        with st.spinner('training in progress...'):
            model.train()
        st.success('Done!')
        st.session_state['train_state'] = True

    st.write('#### Model performance on test set')
    model.evaluate()
    model.save_log()

    st.write('#### Model output visualisation')
    if selected_model == 'Cox Proportional Hazards':
        selected_covariates = st.multiselect(
            'Plot partial effects on outcome for following:',
            model.x.columns.drop('Event_observed')
            )
        model.visualize(selected_covariates)  # pylint: disable=method overrided in cox_ph model
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
