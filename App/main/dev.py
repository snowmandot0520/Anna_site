# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, tabulate_module_summary

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from main.models.beta_BN import BayesianNetwork
from main.models.beta_GP import GaussianProcess

# MAIN SCRIPT --------------------------------------------------------------
def dev(file, verbose):

    @st.cache(allow_output_mutation=True)
    def process_options(model, selected_features, selected_label, censoring):
        model.get_input_features(selected_features)
        model.get_label_feature(label_feature=selected_label, censoring=censoring)
        return model

    st.write('Gaussian Process')

    selected_label = 'Survival time [days]'
    censor_state = st.checkbox('Select to activate left censoring to 1 year')

    ############################################################## 
    # Create RegressionTree class and process options
    ############################################################## 
    model = GaussianProcess()
    model.get_data(file)

    available_input_options = model.get_input_options()  # return value is equal given same df for all subclass of BaseModel
    selected_features = st.multiselect('Input features (select at least one):', available_input_options)
    model = process_options(model, selected_features, selected_label, censor_state)

    if len(model.input_features) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    # Preprocess data
    st.write("### Dataset")
    model.process_input_options(verbose=False)

    st.write("### Train / Test split")
    test_proportion = st.slider('Test set proprotion?', min_value=0.1, max_value=0.5, step=0.1)
    model.train_test_split(test_proportion=test_proportion)

    dev_gp_gpflow(model)

def dev_gp_gpflow(model):
            
    # generate toy data
    np.random.seed(1)
    X = np.random.rand(20, 1)
    Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(20, 1) * 0.01

    m = gpflow.models.GPR((X, Y), kernel=gpflow.kernels.Matern32() + gpflow.kernels.Linear())

    st.write(tabulate_module_summary(m))
    print_summary(m.likelihood)
    m.kernel.kernels[0].lengthscales.assign(0.5)
    m.likelihood.variance.assign(0.01)
    print_summary(m)

    m.trainable_parameters
    p = m.kernel.kernels[0].lengthscales
    p.unconstrained_variable
    p.transform.inverse(p)

    old_parameter = m.kernel.kernels[0].lengthscales
    new_parameter = gpflow.Parameter(
        old_parameter,
        trainable=old_parameter.trainable,
        prior=old_parameter.prior,
        name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
        transform=tfp.bijectors.Exp(),
    )
    m.kernel.kernels[0].lengthscales = new_parameter

    p.transform.inverse(p)

    p = m.kernel.kernels[0].variance
    m.kernel.kernels[0].variance = gpflow.Parameter(p.numpy(), transform=tfp.bijectors.Exp())

    print_summary(m)


def dev_gp_scikitlearn(model):
    
    ############################################################## 
    # GP TUTORAIL 
    ############################################################## 
    # Generate true distribution 
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X))

    # Create training data
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]

    # Create kernel
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    # Fit GP
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)
    gaussian_process.kernel_

    # Run inference
    mean_prediction_tutorial, std_prediction_tutorial = gaussian_process.predict(X, return_std=True)

    # Plot GP Tutorial data
    fig_tutorial, (ax1_tutorial, ax2_tutorial) = plt.subplots(2, 1, figsize=(12, 8))
    fig_tutorial.suptitle('Gaussian Process Tutorial')

    ax1_tutorial.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    ax1_tutorial.set_ylabel("$x$")
    ax1_tutorial.set_xlabel("$f(x)$")
    ax1_tutorial.legend()

    ax2_tutorial.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    ax2_tutorial.scatter(X_train, y_train, label="Observations")
    ax2_tutorial.plot(X, mean_prediction_tutorial, label="Mean prediction")
    ax2_tutorial.fill_between(
        X.ravel(),
        mean_prediction_tutorial - 1.96 * std_prediction_tutorial,
        mean_prediction_tutorial + 1.96 * std_prediction_tutorial,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    ax2_tutorial.legend()
    ax2_tutorial.set_xlabel("$x$")
    ax2_tutorial.set_ylabel("$f(x)$")

    st.pyplot(fig_tutorial)

    ############################################################## 
    # Transplant DATASET
    ############################################################## 

    # Generate true distribution 
    X = model.x_test.drop
    y = model.y_test

    # Create kernel
    #kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    kernel = 1.0 * ExpSineSquared(
        length_scale=1.0,
        periodicity=3.0,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(1.0, 10.0),
    )

    st.write("kernel created")
    st.write(kernel)

    # Fit GP
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
    st.write("GP created")

    with st.spinner("GP Fitting in process"):
        gaussian_process.fit(model.x_train, model.y_train)
    st.success("Fit Done")

    # Run inference
    mean_prediction, std_prediction = gaussian_process.predict(model.x_test, return_std=True)

    # Plot Transplant data
    selected_covariates = model.x.columns
    fig_multi= plt.figure(figsize=(8, 4*len(selected_covariates)))
    gs = gridspec.GridSpec(len(selected_covariates), 1)

    for i, covariate in enumerate(selected_covariates):

        # sort data
        mean_prediction_df = pd.Series(mean_prediction, name='mean_prediction')
        dataframe = pd.concat([model.x_test, model.y_test, mean_prediction_df], axis=1)
        sorted_dataframe = dataframe.sort_values(by=covariate)
        st.write(sorted_dataframe.columns)

        X = sorted_dataframe.drop(columns=['Graft survival censored', 'mean_prediction'])
        Y = sorted_dataframe['Graft survival censored']
        Y_pred = sorted_dataframe['mean_prediction']

        # create axis
        ax = fig_multi.add_subplot(gs[i])
        ax.set_title(covariate)
        ax.plot(X, Y, label="Test_dist", linestyle="dotted")
        ax.scatter(model.x_train[covariate], model.y_train, label="Observations")
        ax.plot(X, Y_pred, label="Mean prediction")
        ax.legend()

    gs.tight_layout(fig_multi)
    st.pyplot(fig_multi)


def dev_bayesian(file):

    @st.cache(allow_output_mutation=True)
    def process_options(model, selected_features, selected_label):
        model.get_input_features(selected_features)
        model.get_label_feature(label_feature=selected_label)
        return model

    st.write('Bayesian Network learning')

    model = BayesianNetwork()

    model.get_data(file)
    available_input_options = model.get_input_options()

    selected_features = st.sidebar.multiselect('Input features (select at least one):', available_input_options)
    selected_label = st.sidebar.selectbox('Prediction target:', ['Graft failure within 1 year'])

    model = process_options(model, selected_features, selected_label)

    while st.button('Run model') == False:
        st.stop()

    model.process_input_options(verbose=True)

    model.train()
    model.visualize()    