"""Functions used commonly across different modes of the app."""
# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st

from app.components.models import support_vector_machine_model
from app.components.models import random_forest_model
from app.components.models import cart_model
from app.components.models import multi_layer_perceptron_model

# FUNCTIONS ----------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def process_options(model, selected_features, selected_label, censoring, duration, rfe):
    model.get_input_features(selected_features)
    model.create_label_feature(label_feature=selected_label, censoring=censoring, duration=duration)
    model.rfe = rfe
    return model

def create_select_label_widget():
    selected_label = st.selectbox('Prediction target:', ['Survival time [days]', 'Failure within given duration [y/n]'])
    if selected_label == 'Failure within given duration [y/n]':
        duration = st.number_input('Failed within [days]', min_value=1, max_value=365, format='%i', value=365,
            help='This input will be used to create boolean labels for classification models')
        censor_state = False
    elif selected_label == 'Survival time [days]':
        censor_state = st.checkbox('Right censoring', value=True)
        if censor_state is True:
            duration = st.number_input('Censor duration [days]', min_value=1, max_value=365, format='%i', value=365,
                help='This input will be used for right censoring of labels for regression models')
        else:
            duration = 1000
    return {'selected_label':selected_label, 'duration':duration, 'censor_state':censor_state}

def create_class_instances(task, model_list):
    model_dict={}
    if task=='classification':
        if 'SVM' in model_list:
            model_dict['SVM'] = support_vector_machine_model.SupportVectorClassifier()
        if 'RF' in model_list:
            model_dict['RF'] = random_forest_model.RandomForest()
        if 'MLP' in model_list:
            mlp_model = multi_layer_perceptron_model.MultiLayerPerceptronClassifier()
            model_dict['MLP'] = mlp_model
    elif task=='regression':
        if 'SVM' in model_list:
            model_dict['SVM'] = support_vector_machine_model.SupportVectorRegression()
        if 'CART' in model_list:
            model_dict['CART'] = cart_model.RegressionTree()
        if 'MLP' in model_list:
            model_dict['MLP'] = multi_layer_perceptron_model.MultiLayerPerceptronRegressor()
    return model_dict

def create_multi_model_parameter_selection(model_dict):
    iterable_params_dict = {}
    for model_name, model_instance in model_dict.items():
        if model_name == 'SVM':
            st.subheader('SVM parameters')
            iterable_params_dict['SVM'] = svm_parameter_widget(model_instance)
        elif model_name == 'RF':
            st.subheader('RF parameters')
            iterable_params_dict['RF'] = rf_parameter_widget(model_instance)
        elif model_name == 'CART':
            st.subheader('CART parameters')
            iterable_params_dict['CART'] = cart_parameter_widget(model_instance)
        elif model_name == 'MLP':
            st.subheader('MLP parameters')
            iterable_params_dict['MLP'] = mlp_parameter_widget(model_instance)
    return iterable_params_dict

def svm_parameter_widget(svm_model):
    c_param = st.select_slider(
        'C parameter:',
        options=svm_model.c_param_list,
        value=(0.01, 1),
        help=('The C parameter in a support vector machine is a regularisation parameter.'
            'The strength of the regularization is inversely proportional to C.')
        )
    kernel = st.multiselect('Kernel:', ['linear', 'poly', 'rbf', 'sigmoid'], default=['linear'],
                            help='Recursive feature elimination function only works for linear kernels.')
    iterables = svm_model.get_iterable_model_options(kernel=kernel, c_param=c_param)
    return iterables

def rf_parameter_widget(rf_model):
    n_estimators = st.select_slider('Number of trees:', options=rf_model.n_estimators_list, value=(10, 50))
    max_depth = st.select_slider('Depth of trees:', options=rf_model.max_depth_list, value=(3, 5))
    criterion = st.multiselect('Criterion?', ['gini', 'entropy'], default='gini')
    iterables = rf_model.get_iterable_model_options(
        criterion=criterion,
        n_estimators=n_estimators,
        max_depth=max_depth)
    return iterables

def cart_parameter_widget(cart_model):
    max_depth = st.select_slider('Depth of tree:', options=cart_model.max_depth_list, value=(3, 5))
    max_leaf_nodes = st.select_slider(
        'Number of leaves:',
        options=cart_model.max_leaf_nodes_list,
        value=(3, 5)
        )
    criterion = st.multiselect(
        'Criterion?',
        ['mse','mae','friedman_mse','poisson'],
        default='mse')
    iterables = cart_model.get_iterable_model_options(
        criterion=criterion,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes
        )
    return iterables

def mlp_parameter_widget(mlp_model):
    hidden_layer_dimensions=[]
    number_of_layers = st.number_input(
        'Hidden layer depth?',
        min_value=1, max_value=3, value=1,
        help='Maximum of 3 hidden layers given computational resource limit')
    for hidden_layer in range(0, number_of_layers):
        layer_size = st.number_input(f'size of layer {hidden_layer}?', min_value=1, max_value=100, value=10)
        hidden_layer_dimensions.append(layer_size)
    mlp_model.hidden_layer_sizes = tuple(hidden_layer_dimensions)
    activation = st.multiselect('Non linear activation?', options=mlp_model.activation_list, default=mlp_model.activation_list[0])
    solver = st.multiselect('Optimizer?', options=mlp_model.solver_list, default=mlp_model.solver_list[0])
    alpha = st.select_slider('Alpha?', options=mlp_model.alpha_list, value=(0.001, 0.003))
    iterables = mlp_model.get_iterable_model_options(
        activation=activation,
        alpha=alpha,
        solver=solver
        )
    return iterables

def multi_model_parameter_iterations(model_dict):
    for model_name, model_instance in model_dict.items():
        if model_name == 'SVM':
            st.subheader('SVM parameters')
            svm_parameter_widget(model_instance)
        elif model_name == 'RF':
            st.subheader('RF parameters')
            rf_parameter_widget(model_instance)
        elif model_name == 'CART':
            st.subheader('CART parameters')
            cart_parameter_widget(model_instance)
        elif model_name == 'MLP':
            st.subheader('MLP parameters')
            mlp_parameter_widget(model_instance)

def set_model_params(model_name, model_instance, params, param_index_dict):
    if model_name == 'SVM':
        model_instance.kernel = params[param_index_dict['kernel']]
        model_instance.c_param = params[param_index_dict['c_param']]
    elif model_name == 'RF':
        model_instance.criterion = params[param_index_dict['criterion']]
        model_instance.max_depth = params[param_index_dict['max_depth']]
        model_instance.n_estimators = params[param_index_dict['n_estimators']]
    elif model_name == 'CART':
        model_instance.criterion = params[param_index_dict['criterion']]
        model_instance.max_depth = params[param_index_dict['max_depth']]
        model_instance.max_leaf_nodes = params[param_index_dict['max_leaf_nodes']]
    elif model_name == 'MLP':
        model_instance.activation = params[param_index_dict['activation']]
        model_instance.alpha = params[param_index_dict['alpha']]
