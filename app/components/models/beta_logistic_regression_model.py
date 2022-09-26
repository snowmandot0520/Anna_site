"""Implements basic logistic regression model"""
# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import streamlit as st

from app.components.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_validate

# DEFINE MODEL -------------------------------------------------------------
class LogisticRegressionClassifier(BaseModel):
    """Implements basic logistic regression model"""
    def __init__(self):
        super().__init__()
        self.model_name = 'Logistic Regression'

    def build_estimator(self):
        self.estimator = LogisticRegression(random_state=0)

    def train(self, k_fold = 5):
        k_fold_cm = cross_validate(
            self.estimator, X=self.x_train, y=self.y_train,
            scoring=['balanced_accuracy', 'roc_auc'],
            cv=k_fold,
            return_train_score=True, return_estimator=True
            )
        self.train_balanced_acc = np.mean(k_fold_cm['train_balanced_accuracy'])
        self.train_roc_auc = np.mean(k_fold_cm['train_roc_auc'])
        self.val_balanced_acc = np.mean(k_fold_cm['test_balanced_accuracy'])
        self.val_roc_auc = np.mean(k_fold_cm['test_roc_auc'])
        # Select best parameters
        validation_performance = k_fold_cm['test_roc_auc']
        self.best_estimator = k_fold_cm['estimator'][np.argmax(validation_performance)]

        if self.verbose:
            st.text(f'{k_fold}-fold train performance: balanced_accuracy = {self.train_balanced_acc:.3f} | '
                    f'ROC AUC = {self.train_roc_auc:.3f}')
            st.text(f'{k_fold}-fold validation performance: balanced_accuracy = {self.val_balanced_acc:.3f} | '
                    f'ROC AUC = {self.val_roc_auc:.3f}')

    def evaluate(self):
        self.y_train_pred = self.best_estimator.predict(self.x_train)
        self.y_train_pred_proba = self.best_estimator.predict_proba(self.x_train)[:,1]
        self.train_mse = mean_squared_error(y_true=self.y_train, y_pred=self.y_train_pred, squared=False)
        self.train_balanced_acc = balanced_accuracy_score(y_true=self.y_train, y_pred=self.y_train_pred)
        self.train_roc_auc = roc_auc_score(y_true=self.y_train, y_score=self.y_train_pred_proba)

        self.y_test_pred = self.best_estimator.predict(self.x_test)
        self.y_test_pred_proba = self.best_estimator.predict_proba(self.x_test)[:,1]
        self.test_mse = mean_squared_error(y_true=self.y_test, y_pred=self.y_test_pred, squared=False)
        self.test_roc_auc = roc_auc_score(y_true=self.y_test, y_score=self.y_test_pred_proba)
        self.test_balanced_acc = balanced_accuracy_score(y_true=self.y_test, y_pred=self.y_test_pred)
        self.test_f1 = f1_score(y_true=self.y_test, y_pred=self.y_test_pred, average='weighted')

        if self.verbose:
            st.text(f'{self.model_name} test performance: '
                    f'balanced_accuracy = {self.test_balanced_acc:.3f} | Weighted F1 = {self.test_f1:.3f} | '
                    f'ROC_AUC = {self.test_roc_auc:.3f}')

    def visualize(self):
        with st.expander('Confusion matrix'):
            with st.spinner('creating image...'):
                self.plot_confusion_matrix()

        with st.expander('Variable importance'):
            with st.spinner('creating image...'):
                self.plot_variable_importance()

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': str(self.input_features),
            'label_feature': self.label_feature,
            'train_mse':self.train_mse, 'train_roc_auc':self.train_roc_auc,
            'train_balanced_acc':self.train_balanced_acc,
            'val_balanced_acc':self.val_balanced_acc, 'val_roc_auc':self.val_roc_auc,
            'test_mse':self.test_mse, 'test_roc_auc':self.test_roc_auc,
            'test_f1':self.test_f1, 'test_balanced_acc':self.test_balanced_acc}
        if self.verbose:
            print(f'saving log: {cache}')
        self.log.append(cache)

    def save_fig(self):
        self.fig_list = [self.confusion_matrix_plot, self.variable_importance_plot]
