# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

from main.models.BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import RFE


# DEFINE MODEL -------------------------------------------------------------
class RandomForest(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = "Random Forest Classifier"
        self.n_estimators = 100
        self.n_estimators_list = [1, 10, 50, 100, 200, 500, 1000, 10000]
        self.max_depth = 4
        self.max_depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.criterion = 'gini'
        self.class_weight = 'balanced_subsample'
        self.n_features_to_select = 1
        self.rfe_step = 1
        self.iterable_model_options_dict = {'n_estimators':self.n_estimators, 'n_estimators_list':self.n_estimators_list,
                                            'max_depth':self.max_depth, 'max_depth_list':self.max_depth_list}

    def build_classifier(self):
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                 criterion=self.criterion, class_weight=self.class_weight)

    def train(self, K_fold = 5, verbose=False):

        if self.rfe:
            selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select, step=self.rfe_step)
            self.best_classifier = selector.fit(self.X_train, self.Y_train)
            self.sort_feature_importance()
        else:
            # K-fold cross validation
            k_fold_cm = cross_validate(self.classifier, X=self.X_train, y=self.Y_train, scoring=['accuracy', 'roc_auc'], cv=K_fold,
                                       return_train_score=True, return_estimator=True)
            self.train_acc, self.train_roc_acu = np.mean(k_fold_cm['train_accuracy']), np.mean(k_fold_cm['train_roc_auc'])
            self.val_acc, self.val_roc_auc = np.mean(k_fold_cm['test_accuracy']), np.mean(k_fold_cm['test_roc_auc'])

            if verbose:
                st.text("{}-fold train performance: Accuracy = {:.3f} | ROC AUC = {:.3f}".format(K_fold, self.train_acc, self.train_roc_acu))
                st.text("{}-fold validation performance: Accuracy = {:.3f} | ROC AUC = {:.3f}".format(K_fold, self.val_acc, self.val_roc_auc))

            # Select best parameters
            validation_performance = k_fold_cm['test_roc_auc']
            self.best_classifier = k_fold_cm['estimator'][np.argmax(validation_performance)]

    def evaluate(self, verbose=False):
        self.Y_train_pred = self.best_classifier.predict(self.X_train)
        self.Y_test_pred = self.best_classifier.predict(self.X_test)

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
                 'class_weight': self.class_weight, 'n_estimator': self.n_estimators, 'max_depth': self.max_depth, 'criterion':self.criterion,
                 'train_acc':self.train_acc, 'train_roc_acu':self.train_roc_acu, 'val_acc':self.val_acc, 'val_roc_auc':self.val_roc_auc,
                 'test_acc':self.test_acc, 'test_f1':self.test_f1}
        if self.rfe:
            cache.update({'features_sorted_by_importance':self.sorted_features})
        self.log = pd.DataFrame(data=cache)
