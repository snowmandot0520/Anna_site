"""Runs experiment mode where you can train iteratively over different hyperparameter settings"""
# TODO: implement iterative option for MLP layer size.

# LOAD DEPENDENCY ----------------------------------------------------------
from time import time
import streamlit as st
import time

from itertools import product
from app.components.utils import widget_functions


# MAIN SCRIPT --------------------------------------------------------------
def experiment(file, verbose):
    st.header('Experiment setting')
    label_info = widget_functions.create_select_label_widget()

    if label_info['selected_label'] == 'Failure within given duration [y/n]':
        # Binary classification task
        model_list = st.multiselect('Models to train in this session:', ['SVM', 'RF', 'MLP'])
        model_dict = widget_functions.create_class_instances(task='classification', model_list=model_list)

    elif label_info['selected_label'] == 'Survival time [days]':
        # Regression task
        model_list = st.multiselect('Models to train in this session:', ['SVM', 'CART', 'MLP'])
        model_dict = widget_functions.create_class_instances(task='regression', model_list=model_list)

    # run following if and only if at least one model type has been selected
    if len(model_dict) == 0:
        st.stop()
    # create hyperparameter selection widgets
    iterable_params_dict = widget_functions.create_multi_model_parameter_selection(model_dict)

    st.subheader('Feature selection')
    for model_name, model_instance in model_dict.items():
        model_instance.get_data(file)
        model_instance.verbose = verbose
        # return value is equal given same df for all subclass of BaseModel
        available_input_options = model_instance.get_input_options()

    selected_features = st.multiselect(
        'Input features (select at least one):',
        available_input_options,
        default=available_input_options
        )
    feature_elimination = st.checkbox(
        'Recursive feature elimination?',
        value=False,
        help='Must select at least 2 features'
        )

    if len(selected_features) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    for model_name, model_instance in model_dict.items():
        model_dict[model_name] = widget_functions.process_options(
            model=model_dict[model_name],
            selected_features=selected_features,
            selected_label=label_info['selected_label'],
            censoring=label_info['censor_state'],
            duration=label_info['duration'],
            rfe=feature_elimination
            )
        subset_dict = model_instance.get_subset_options_dict()
        model_instance.create_dataframe_subset(subset_dict)

    for model_name, model_instance in model_dict.items():
        model_instance.process_input_options()

    st.write('### Train / Test split')
    test_proportion = st.slider('Test set proprotion?', min_value=0.1, max_value=0.5, step=0.1, value=0.3)
    for model_name, model_instance in model_dict.items():
        model_instance.train_test_split(test_proportion=test_proportion)

    # Overwrite command line verbose option for experiment setting
    print_status = st.checkbox('Print training performance')
    for model_name, model_instance in model_dict.items():
        model_instance.verbose = print_status

    if st.session_state['train_state'] is False:
        st.session_state['continue_state'] = st.button('Continue')
    if st.session_state['continue_state'] is False:
        st.stop()

    while st.session_state['train_state'] is False:

        with st.spinner('training in progress...'):

            for model_name, model_params_iterables in iterable_params_dict.items():

                if verbose:
                    print(f'Running iterations for {model_name}')
                    model_start_time = time.time()

                # change dictionary to iterable list form using dot product
                collected_param_values = []
                param_index_dict = {}
                for param_name, param_val_list in model_params_iterables.items():
                    collected_param_values.append(param_val_list)
                    param_index_dict[param_name] = len(collected_param_values)-1

                # run iterations of all possible parameter sets
                for params in list(product(*collected_param_values)):
                    if verbose:
                        print(f'running parameter set: {params}')
                        start_time = time.time()
                    model_instance = model_dict[model_name]
                    widget_functions.set_model_params(model_name, model_instance, params, param_index_dict)

                    model_instance.build_estimator()
                    model_instance.train()
                    model_instance.evaluate()
                    model_instance.save_log()

                    if model_instance.label_feature[-5:] == '[y/n]':
                        model_instance.store_best_estimator(metric='roc_auc')
                        model_instance.store_estimator_plots()
                    if verbose:
                        end_time = time.time()
                        print(f'time taken: {(end_time-start_time):.3f}s')

                if verbose:
                    model_end_time = time.time()
                    print(f'total time taken for {model_name}: {(model_end_time-model_start_time):.3f}s')

        st.success('Done!')
        st.session_state['train_state'] = True

    st.write('#### Save options')
    # create download option from browser
    for model_name, model_instance in model_dict.items():
        model_instance.create_zip_download_button()

    train_again = st.button('Train again')
    if train_again:
        st.session_state['train_state'] = False
