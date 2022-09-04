"""Implements random forest model for classification.
Refer to CART for random forest based regression."""
# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import streamlit as st

from app.components.models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE


# DEFINE MODEL -------------------------------------------------------------
class RandomForest(BaseModel):
    """Implements random forest classifier"""
    def __init__(self):
        super().__init__()
        self.model_name = 'Random Forest Classifier'
        self.n_estimators = 100
        self.n_estimators_list = [1, 3, 5, 10, 30, 50, 100, 200, 500, 1000]
        self.max_depth = 4
        self.max_depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.criterion = 'gini'
        self.class_weight = 'balanced_subsample'
        self.n_features_to_select = 1
        self.rfe_step = 1
        self.iterable_model_options_dict = {
            'n_estimators':self.n_estimators, 'n_estimators_list':self.n_estimators_list,
            'max_depth':self.max_depth, 'max_depth_list':self.max_depth_list
            }
        self.option_widget_type_dict = {
            'criterion':'unordered',
            'n_estimators':'ordered',
            'max_depth':'ordered'
            }

    def build_estimator(self):
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            criterion=self.criterion, class_weight=self.class_weight
            )

    def train(self, k_fold = 5):
        if self.rfe:
            self.selector = RFE(
                self.estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.rfe_step)
            self.best_estimator = self.selector.fit(self.x_train, self.y_train)
            self.sort_feature_importance()
            self.train_acc, self.train_roc_auc, self.val_acc, self.val_roc_auc = np.NaN, np.NaN, np.NaN, np.NaN
        else:
            # K-fold cross validation
            k_fold_cm = cross_validate(
                self.estimator, X=self.x_train, y=self.y_train,
                scoring=['accuracy', 'roc_auc'], cv=k_fold,
                return_train_score=True, return_estimator=True
                )
            self.train_acc = np.mean(k_fold_cm['train_accuracy'])
            self.train_roc_auc = np.mean(k_fold_cm['train_roc_auc'])
            self.val_acc = np.mean(k_fold_cm['test_accuracy'])
            self.val_roc_auc = np.mean(k_fold_cm['test_roc_auc'])

            if self.verbose:
                st.text(f'{k_fold}-fold train performance: Accuracy = {self.train_acc:.3f} | '
                        f'ROC AUC = {self.train_roc_auc:.3f}')
                st.text(f'{k_fold}-fold validation performance: Accuracy = {self.val_acc:.3f} | '
                        f'ROC AUC = {self.val_roc_auc:.3f}')

            # Select best parameters
            validation_performance = k_fold_cm['test_roc_auc']
            self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

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
        cache = {
            'model': self.model_name, 'input_features': str(self.input_features),
            'label_feature': self.label_feature, 'class_weight': self.class_weight,
            'n_estimator': self.n_estimators, 'max_depth': self.max_depth,
            'criterion':self.criterion,
            'train_mse':self.train_mse, 'train_roc_auc':self.train_roc_auc,
            'val_acc':self.val_acc, 'val_roc_auc':self.val_roc_auc,
            'test_mse':self.test_acc, 'test_roc_auc':self.test_roc_auc,
            'test_f1':self.test_f1, 'test_acc':self.test_acc}
        if self.rfe:
            cache.update({'features_sorted_by_importance':self.sorted_features})
        if self.verbose:
            print(f'saving log: {cache}')
        self.log.append(cache)

    def save_fig(self):
        self.fig_list = [self.confusion_matrix_plot, self.variable_importance_plot]
