# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import streamlit as st

from app.components.models.base_model import BaseModel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
from sklearn.model_selection import cross_validate
from lifelines.utils import concordance_index

# DEFINE MODEL -------------------------------------------------------------
class MultiLayerPerceptron(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = 'Multi-layer Perceptron Regressor'
        self.hidden_layer_sizes = (100,)
        self.activation = 'relu'
        self.alpha = 0.001

    def build_estimator(self):
        self.estimator = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                                      alpha=self.alpha)

    def train(self, K_fold = 5, verbose=False):

        # K-fold cross validation
        k_fold_cm = cross_validate(self.estimator, X=self.x_train, y=self.y_train, scoring=['neg_root_mean_squared_error', 'r2'], cv=K_fold,
                                   return_train_score=True, return_estimator=True)
        self.train_acc, self.train_r2 = np.mean(k_fold_cm['train_neg_root_mean_squared_error']), np.mean(k_fold_cm['train_r2'])
        self.val_acc, self.val_r2 = np.mean(k_fold_cm['test_neg_root_mean_squared_error']), np.mean(k_fold_cm['test_r2'])

        if verbose:
            st.text('{}-fold train performance: RMSE = {:.3f} | R^2 = {:.3f}'.format(K_fold, self.train_acc, self.train_r2))
            st.text('{}-fold validation performance: RMSE = {:.3f} | R^2 = {:.3f}'.format(K_fold, self.val_acc, self.val_r2))

        # Select best parameters
        validation_performance = k_fold_cm['test_neg_root_mean_squared_error']
        self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

    def evaluate(self, verbose=False):
        self.y_train_pred = self.best_estimator.predict(self.x_train)
        self.y_test_pred = self.best_estimator.predict(self.x_test)

        self.test_acc = mean_squared_error(y_true=self.y_test, y_pred=self.y_test_pred, squared=False)
        self.test_r2 = r2_score(y_true=self.y_test, y_pred=self.y_test_pred)
        if verbose:
            st.text('{} test performance: RMSE = {:.3f} | R^2 = {:.3f}'.format(self.model_name, self.test_acc, self.test_r2))

        self.train_ci = concordance_index(event_times=self.y_train, predicted_scores=self.y_train_pred)
        self.test_ci = concordance_index(event_times=self.y_test, predicted_scores=self.y_test_pred)

    def visualize(self):
        with st.beta_expander('Confusion matrix'):
            st.write('To be implemented')

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': self.input_features, 'label_feature': self.label_feature,
                 'train_acc':self.train_acc, 'train_r2':self.train_r2, 'val_acc':self.val_acc, 'val_r2':self.val_r2,
                 'test_acc':self.test_acc, 'test_r2':self.test_r2,
                 'train_ci':self.train_ci, 'test_ci':self.test_ci}
        self.log.append(cache)
