"""Implements random forest model for classification.
Refer to CART for random forest based regression."""
# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

from app.components.models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE


# DEFINE MODEL -------------------------------------------------------------
class RandomForest(BaseModel):
    """Implements random forest classifier"""
    def __init__(self):
        super().__init__()
        self.model_name = 'Random Forest Classifier'
        self.n_estimators = 100
        self.n_estimators_list = [1, 10, 50, 100, 200, 500, 1000, 10000]
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

    def build_classifier(self):
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            criterion=self.criterion, class_weight=self.class_weight
            )

    def train(self, k_fold = 5, verbose=False):
        if self.rfe:
            selector = RFE(
                self.estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.rfe_step)
            self.best_classifier = selector.fit(self.x_train, self.y_train)
            self.sort_feature_importance()
        else:
            # K-fold cross validation
            k_fold_cm = cross_validate(
                self.classifier, X=self.x_train, y=self.y_train,
                scoring=['accuracy', 'roc_auc'], cv=k_fold,
                return_train_score=True, return_estimator=True
                )
            self.train_acc = np.mean(k_fold_cm['train_accuracy'])
            self.train_roc_auc = np.mean(k_fold_cm['train_roc_auc'])
            self.val_acc = np.mean(k_fold_cm['test_accuracy'])
            self.val_roc_auc = np.mean(k_fold_cm['test_roc_auc'])

            if verbose:
                st.text(f'{k_fold}-fold train performance: Accuracy = {self.train_acc:.3f} | '
                        f'ROC AUC = {self.train_roc_auc:.3f}')
                st.text(f'{k_fold}-fold validation performance: Accuracy = {self.val_acc:.3f} | '
                        f'ROC AUC = {self.val_roc_auc:.3f}')

            # Select best parameters
            validation_performance = k_fold_cm['test_roc_auc']
            self.best_classifier = k_fold_cm['estimator'][np.argmax(validation_performance)]

    def evaluate(self, verbose=False):
        self.y_train_pred = self.best_classifier.predict(self.x_train)
        self.y_test_pred = self.best_classifier.predict(self.x_test)

        self.test_acc = accuracy_score(y_true=self.y_test, y_pred=self.y_test_pred)
        self.test_f1 = f1_score(y_true=self.y_test, y_pred=self.y_test_pred, average='weighted')
        if verbose:
            st.text(f'{self.model_name} test performance: Accuracy = {self.test_acc:.3f}'
                    f' | Weighted F1 = {self.test_f1:.3f}')

    def visualize(self):
        with st.expander('Confusion matrix'):
            with st.spinner('creating image...'):
                self.plot_confusion_matrix()

        with st.expander('Variable importance'):
            with st.spinner('creating image...'):
                self.plot_variable_importance()

    def save_log(self):
        cache = {
            'model': self.model_name, 'input_features': self.input_features,
            'label_feature': self.label_feature, 'class_weight': self.class_weight,
            'n_estimator': self.n_estimators, 'max_depth': self.max_depth,
            'criterion':self.criterion, 'train_acc':self.train_acc,
            'train_roc_acu':self.train_roc_acu, 'val_acc':self.val_acc,
            'val_roc_auc':self.val_roc_auc, 'test_acc':self.test_acc,
            'test_f1':self.test_f1
            }
        if self.rfe:
            cache.update({'features_sorted_by_importance':self.sorted_features})
        self.log = pd.DataFrame(data=cache)
