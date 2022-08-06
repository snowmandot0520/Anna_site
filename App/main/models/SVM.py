# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import streamlit as st
import pandas as pd

from main.models.BaseModel import BaseModel
from main.utils.RFECV import RFECV
from sklearn.svm import SVC, SVR
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import RFE
from lifelines.utils import concordance_index

# CLASS  OBJECT -----------------------------------------------------------
class SupportVectorClassifier(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = "Support Vector Classifier"
        self.class_weight = 'balanced'
        self.c_param = 1
        self.c_param_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        self.kernel = 'linear'
        self.n_features_to_select = 1
        self.rfe_step = 1
        self.iterable_model_options_dict = {'c_param':self.c_param, 'c_param_list':self.c_param_list}

    def build_estimator(self):
        self.estimator = SVC(probability=True, C=self.c_param, kernel=self.kernel, class_weight=self.class_weight.lower())

    def train(self, K_fold = 5, verbose=False):
        # Recursive feature elimination
        if self.rfe == True:
            self.selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select, step=self.rfe_step, verbose=1)
            self.best_estimator = self.selector.fit(self.X_train, self.Y_train)
            self.sort_feature_importance()
            self.train_acc, self.train_r2, self.val_acc, self.val_r2 = np.NaN, np.NaN, np.NaN, np.NaN

        else:
            # K-fold cross validation
            k_fold_cm = cross_validate(self.estimator, X=self.X_train, y=self.Y_train, scoring=['accuracy', 'roc_auc'], cv=K_fold,
                                       return_train_score=True, return_estimator=True)
            self.train_acc, self.train_roc_acu = np.mean(k_fold_cm['train_accuracy']), np.mean(k_fold_cm['train_roc_auc'])
            self.val_acc, self.val_roc_auc = np.mean(k_fold_cm['test_accuracy']), np.mean(k_fold_cm['test_roc_auc'])

            if verbose:
                st.text("{}-fold train performance: Accuracy = {:.3f} | ROC AUC = {:.3f}".format(K_fold, self.train_acc, self.train_roc_acu))
                st.text("{}-fold validation performance: Accuracy = {:.3f} | ROC AUC = {:.3f}".format(K_fold, self.val_acc, self.val_roc_auc))

            # Select best parameters
            validation_performance = k_fold_cm['test_roc_auc']
            self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

    def evaluate(self, verbose=False):
        self.Y_train_pred = self.best_estimator.predict(self.X_train)
        self.Y_test_pred = self.best_estimator.predict(self.X_test)

        self.test_acc = accuracy_score(y_true=self.Y_test, y_pred=self.Y_test_pred)
        self.test_f1 = f1_score(y_true=self.Y_test, y_pred=self.Y_test_pred, average='weighted')
        if verbose:
            st.text("{} test performance: Accuracy = {:.3f} | Weighted F1 = {:.3f}".format(self.model_name, self.test_acc, self.test_f1))

    def visualize(self):
        with st.expander('Confusion matrix'):
            with st.spinner('creating image...'):
                self.plot_confusion_matrix()

        with st.expander('Variable importance'):
            with st.spinner('creating image...'):
                self.plot_variable_importance()

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': self.input_features, 'label_feature': self.label_feature,
                 'class_weight': self.class_weight, 'c_param': self.c_param, 'kernel': self.kernel,
                 'train_acc':self.train_acc, 'train_roc_acu':self.train_roc_acu, 'val_acc':self.val_acc, 'val_roc_auc':self.val_roc_auc,
                 'test_acc':self.test_acc, 'test_f1':self.test_f1}
        if self.rfe:
            cache.update({'features_sorted_by_importance':self.sorted_features})
        self.log = pd.DataFrame(data=cache)


class SupportVectorRegression(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = "Support Vector Regression"
        self.c_param = 1
        self.c_param_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        self.kernel = 'linear'
        self.n_features_to_select = 1
        self.rfe_step = 1
        self.iterable_model_options_dict = {'c_param':self.c_param, 'c_param_list':self.c_param_list}

    def build_estimator(self):
        self.estimator = SVR(C=self.c_param, kernel=self.kernel)

    def train(self, K_fold = 5, verbose=False):

        if self.rfe:
            self.selector = RFECV(self.estimator, scoring='neg_root_mean_squared_error', cv=K_fold, n_jobs=1)
            self.best_estimator = self.selector.fit(self.X_train, self.Y_train)
            self.sort_feature_importance()
            self.train_acc, self.train_r2, self.val_acc, self.val_r2 = np.NaN, np.NaN, np.NaN, np.NaN

        else:
            # K-fold cross validation
            k_fold_cm = cross_validate(self.estimator, X=self.X_train, y=self.Y_train, scoring=['neg_root_mean_squared_error', 'r2'], cv=K_fold,
                                       return_train_score=True, return_estimator=True)
            self.train_acc, self.train_r2 = np.mean(k_fold_cm['train_neg_root_mean_squared_error']), np.mean(k_fold_cm['train_r2'])
            self.val_acc, self.val_r2 = np.mean(k_fold_cm['test_neg_root_mean_squared_error']), np.mean(k_fold_cm['test_r2'])

            if verbose:
                st.text("{}-fold train performance: RMSE = {:.3f} | R^2 = {:.3f}".format(K_fold, self.train_acc, self.train_r2))
                st.text("{}-fold validation performance: RMSE = {:.3f} | R^2 = {:.3f}".format(K_fold, self.val_acc, self.val_r2))

            # Select best parameters
            validation_performance = k_fold_cm['test_neg_root_mean_squared_error']
            self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

    def evaluate(self, verbose=False):
        self.Y_train_pred = self.best_estimator.predict(self.X_train)
        self.Y_test_pred = self.best_estimator.predict(self.X_test)

        self.train_acc = mean_squared_error(y_true=self.Y_train, y_pred=self.Y_train_pred, squared=False)
        self.test_acc = mean_squared_error(y_true=self.Y_test, y_pred=self.Y_test_pred, squared=False)
        self.train_r2 = r2_score(y_true=self.Y_train, y_pred=self.Y_train_pred)
        self.test_r2 = r2_score(y_true=self.Y_test, y_pred=self.Y_test_pred)
        self.train_ci = concordance_index(event_times=self.Y_train, predicted_scores=self.Y_train_pred)
        self.test_ci = concordance_index(event_times=self.Y_test, predicted_scores=self.Y_test_pred)

        if verbose:
            st.text("{} train performance: RMSE = {:.3f} | R^2 = {:.3f} | CI = {:.3f}".format(self.model_name, self.train_acc, self.train_r2, self.train_ci))
            st.text("{} test performance: RMSE = {:.3f} | R^2 = {:.3f}".format(self.model_name, self.test_acc, self.test_r2))

    def visualize(self):
        with st.expander('Plot outcome'):

            if self.rfe:
                self.plot_recursive_feature_elimination_cross_validation_test()

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': self.input_features, 'label_feature': self.label_feature,
                 'class_weight': np.NaN, 'c_param': self.c_param, 'kernel': self.kernel,
                 'train_acc':self.train_acc, 'train_r2':self.train_r2, 'val_acc':self.val_acc, 'val_r2':self.val_r2,
                 'test_acc':self.test_acc, 'test_r2':self.test_r2,
                 'train_ci':self.train_ci, 'test_ci':self.test_ci}
        if self.rfe:
            cache.update({'features_sorted_by_importance':self.sorted_features})
        self.log = pd.DataFrame(data=cache)

    def save_fig(self):
        self.fig_list = []
        if self.rfe:
            self.fig_list.append(self.fig_rfecv)

