"""Implements basic MLP model"""
# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

from app.components.models.base_model import BaseModel
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_validate
from lifelines.utils import concordance_index

# DEFINE MODEL -------------------------------------------------------------
class MultiLayerPerceptronClassifier(BaseModel):
    """Implements basic MLP model"""
    def __init__(self):
        super().__init__()
        self.model_name = 'Multi-layer Perceptron Classifier'
        self.hidden_layer_sizes = (100,)
        self.activation = 'relu'
        self.activation_list =  ['relu', 'identity', 'logistic', 'tanh']
        self.alpha = 0.001
        self.alpha_list = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
        self.option_widget_type_dict = {
            'activation':'unordered',
            'alpha':'ordered'
            }
        self.iterable_model_options_dict = {
            'activation':self.activation, 'activation_list':self.activation_list,
            'alpha':self.alpha, 'alpha_list':self.alpha_list
            }

    def build_estimator(self):
        self.estimator = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                                      alpha=self.alpha)

    def train(self, k_fold = 5):
        k_fold_cm = cross_validate(
            self.estimator, X=self.x_train, y=self.y_train,
            scoring=['accuracy', 'roc_auc'],
            cv=k_fold,
            return_train_score=True, return_estimator=True
            )
        self.train_acc = np.mean(k_fold_cm['train_accuracy'])
        self.train_roc_auc = np.mean(k_fold_cm['train_roc_auc'])
        self.val_acc = np.mean(k_fold_cm['test_accuracy'])
        self.val_roc_auc = np.mean(k_fold_cm['test_roc_auc'])
        # Select best parameters
        validation_performance = k_fold_cm['test_roc_auc']
        self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

        if self.verbose:
            st.text(f'{k_fold}-fold train performance: Accuracy = {self.train_acc:.3f} | '
                    f'ROC AUC = {self.train_roc_auc:.3f}')
            st.text(f'{k_fold}-fold validation performance: Accuracy = {self.val_acc:.3f} | '
                    f'ROC AUC = {self.val_roc_auc:.3f}')

    def evaluate(self):
        self.y_train_pred = self.best_estimator.predict(self.x_train)
        self.y_train_pred_proba = self.best_estimator.predict_proba(self.x_train)[:,1]
        self.train_mse = mean_squared_error(y_true=self.y_train, y_pred=self.y_train_pred, squared=False)
        self.train_roc_auc = roc_auc_score(y_true=self.y_train, y_score=self.y_train_pred_proba)

        self.y_test_pred = self.best_estimator.predict(self.x_test)
        self.y_test_pred_proba = self.best_estimator.predict_proba(self.x_test)[:,1]
        self.test_mse = mean_squared_error(y_true=self.y_test, y_pred=self.y_test_pred, squared=False)
        self.test_roc_auc = roc_auc_score(y_true=self.y_test, y_score=self.y_test_pred_proba)
        self.test_acc = accuracy_score(y_true=self.y_test, y_pred=self.y_test_pred)
        self.test_f1 = f1_score(y_true=self.y_test, y_pred=self.y_test_pred, average='weighted')

        if self.verbose:
            st.text(f'{self.model_name} test performance: '
                    f'Accuracy = {self.test_acc:.3f} | Weighted F1 = {self.test_f1:.3f} | '
                    f'ROC_AUC = {self.test_roc_auc:.3f}')

    def visualize(self):
        with st.expander('Confusion matrix'):
            with st.spinner('creating image...'):
                self.plot_confusion_matrix()

        with st.expander('Variable importance'):
            with st.spinner('creating image...'):
                self.plot_variable_importance()

    def save_log(self):
        st.write('tag')
        cache = {'model': self.model_name, 'input_features': str(self.input_features),
            'label_feature': self.label_feature, 'hidden_layer_sizes': str(self.hidden_layer_sizes),
            'activation': self.activation, 'alpha': self.alpha,
            'train_mse':self.train_mse, 'train_roc_auc':self.train_roc_auc,
            'val_acc':self.val_acc, 'val_roc_auc':self.val_roc_auc,
            'test_mse':self.test_acc, 'test_roc_auc':self.test_roc_auc,
            'test_f1':self.test_f1, 'test_acc':self.test_acc}
        if self.verbose:
            print(f'saving log: {cache}')
        self.log.append(cache)

    def save_fig(self):
        self.fig_list = [self.confusion_matrix_plot, self.variable_importance_plot]


class MultiLayerPerceptronRegressor(BaseModel):
    """Implements basic MLP model"""
    def __init__(self):
        super().__init__()
        self.model_name = 'Multi-layer Perceptron Regressor'
        self.hidden_layer_sizes = (100,)
        self.activation = 'relu'
        self.activation_list =  ['relu', 'identity', 'logistic', 'tanh']
        self.alpha = 0.001
        self.alpha_list = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
        self.option_widget_type_dict = {
            'activation':'unordered',
            'alpha':'ordered'
            }
        self.iterable_model_options_dict = {
            'activation':self.activation, 'activation_list':self.activation_list,
            'alpha':self.alpha, 'alpha_list':self.alpha_list
            }

    def build_estimator(self):
        self.estimator = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                                      alpha=self.alpha)

    def train(self, k_fold = 5):
        k_fold_cm = cross_validate(
            self.estimator,
            X=self.x_train,
            y=self.y_train,
            scoring=['neg_root_mean_squared_error', 'r2'],
            cv=k_fold,
            return_train_score=True, return_estimator=True
            )
        self.train_acc = np.mean(k_fold_cm['train_neg_root_mean_squared_error'])
        self.train_r2 = np.mean(k_fold_cm['train_r2'])
        self.val_acc = np.mean(k_fold_cm['test_neg_root_mean_squared_error'])
        self.val_r2 = np.mean(k_fold_cm['test_r2'])
        # Select best parameters
        validation_performance = k_fold_cm['test_neg_root_mean_squared_error']
        self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

        if self.verbose:
            st.text(f'{k_fold}-fold train performance: RMSE = {self.train_acc:.3f} | R^2 = {self.train_r2:.3f}')
            st.text(f'{k_fold}-fold validation performance: RMSE = {self.val_acc:.3f} | R^2 = {self.val_r2:.3f}')

    def evaluate(self):
        self.y_train_pred = self.best_estimator.predict(self.x_train)
        self.y_test_pred = self.best_estimator.predict(self.x_test)
        self.train_acc = mean_squared_error(y_true=self.y_train, y_pred=self.y_train_pred, squared=False)
        self.test_acc = mean_squared_error(y_true=self.y_test, y_pred=self.y_test_pred, squared=False)
        self.train_r2 = r2_score(y_true=self.y_train, y_pred=self.y_train_pred)
        self.test_r2 = r2_score(y_true=self.y_test, y_pred=self.y_test_pred)
        self.train_ci = concordance_index(event_times=self.y_train, predicted_scores=self.y_train_pred)
        self.test_ci = concordance_index(event_times=self.y_test, predicted_scores=self.y_test_pred)

        if self.verbose:
            st.text(f'{self.model_name} train performance: '
                    f'RMSE = {self.train_acc:.3f} | R^2 = {self.train_r2:.3f} | CI = {self.train_ci:.3f}')
            st.text(f'{self.model_name} test performance: '
                    f'RMSE = {self.test_acc:.3f} | R^2 = {self.test_r2:.3f} | CI = {self.test_ci:.3f}')

    def visualize(self):
        with st.expander('Plot outcome'):
            st.write('To be implemented')

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': str(self.input_features),
            'label_feature': self.label_feature, 'hidden_layer_sizes': str(self.hidden_layer_sizes),
            'activation': self.activation, 'alpha': self.alpha,
            'train_acc':self.train_acc, 'train_r2':self.train_r2,
            'val_acc':self.val_acc, 'val_r2':self.val_r2,
            'test_acc':self.test_acc, 'test_r2':self.test_r2,
            'train_ci':self.train_ci, 'test_ci':self.test_ci}
        if self.verbose:
            print(f'saving log: {cache}')
        self.log.append(cache)

    def save_fig(self):
        self.fig_list = []
