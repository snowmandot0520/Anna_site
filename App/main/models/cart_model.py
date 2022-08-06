# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from App.main.models.base_model import BaseModel
from main.utils.RFECV import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import *
from sklearn.model_selection import cross_validate
from lifelines.utils import concordance_index

# DEFINE MODEL -------------------------------------------------------------
class RegressionTree(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = 'Regression Tree'
        self.criterion = 'mse'
        self.max_leaf_nodes = int()
        self.max_leaf_nodes_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.max_depth = int()
        self.max_depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.iterable_model_options_dict = {'max_leaf_nodes':self.max_leaf_nodes, 'max_depth':self.max_depth,
                                            'max_leaf_nodes_list':self.max_leaf_nodes_list, 'max_depth_list':self.max_depth_list}

    def build_estimator(self):
        self.estimator = DecisionTreeRegressor(criterion=self.criterion, splitter='best')

    def train(self, K_fold = 5, verbose=False):

        if self.rfe:
            self.selector = RFECV(self.estimator, scoring='neg_root_mean_squared_error', cv=K_fold, n_jobs=1)
            self.best_estimator = self.selector.fit(self.x_train, self.y_train)
            self.sort_feature_importance()
            self.train_acc, self.train_r2, self.val_acc, self.val_r2 = np.NaN, np.NaN, np.NaN, np.NaN

        else:
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

        self.train_acc = mean_squared_error(y_true=self.y_train, y_pred=self.y_train_pred, squared=False)
        self.test_acc = mean_squared_error(y_true=self.y_test, y_pred=self.y_test_pred, squared=False)
        self.train_r2 = r2_score(y_true=self.y_train, y_pred=self.y_train_pred)
        self.test_r2 = r2_score(y_true=self.y_test, y_pred=self.y_test_pred)
        self.train_ci = concordance_index(event_times=self.y_train, predicted_scores=self.y_train_pred)
        self.test_ci = concordance_index(event_times=self.y_test, predicted_scores=self.y_test_pred)

        if verbose:
            st.text('{} train performance: RMSE = {:.3f} | R^2 = {:.3f} | CI = {:.4f}'.format(self.model_name, self.train_acc, self.train_r2, self.train_ci))
            st.text('{} test performance: RMSE = {:.3f} | R^2 = {:.3f} | CI = {:.4f}'.format(self.model_name, self.test_acc, self.test_r2, self.test_ci))

    def visualize(self):

        with st.expander('Plot data distribution'.format(self.model_name)):

            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(self.y_train, bins=range(0,max(366, int(max(self.y_train)))))
            fig_hist.tight_layout()

            st.pyplot(fig_hist)
            self.fig_hist = fig_hist

        with st.expander('Plot outcome'):
            print(self.rfe)

            if self.rfe:
                self.plot_recursive_feature_elimination_cross_validation_test()
                estimator = self.best_estimator.estimator_
            else:
                estimator = self.best_estimator

            fig_tree, ax_tree = plt.subplots(figsize=(30,20))
            tree.plot_tree(estimator, ax=ax_tree)
            fig_tree.tight_layout()
            fig_tree.suptitle('Tree visualisation')

            st.pyplot(fig_tree)
            self.fig_tree = fig_tree

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': self.input_features, 'label_feature': self.label_feature,
                 'max_depth': self.max_depth, 'max_leaf_nodes': self.max_leaf_nodes,
                 'criterion': self.criterion,
                 'train_acc':self.train_acc, 'train_r2':self.train_r2, 'val_acc':self.val_acc, 'val_r2':self.val_r2,
                 'test_acc':self.test_acc, 'test_r2':self.test_r2,
                 'train_ci':self.train_ci, 'test_ci':self.test_ci}
        if self.rfe:
            cache.update({'features_sorted_by_importance':self.sorted_features})

        print(cache)
        self.log = pd.DataFrame(data=cache)

    def save_fig(self):
        self.fig_list = [self.fig_hist, self.fig_tree]
        if self.rfe:
            self.fig_list.append(self.fig_rfecv)