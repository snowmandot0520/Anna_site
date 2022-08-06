# LOAD DEPENDENCY ----------------------------------------------------------
import os
import io
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from lifelines.utils import concordance_index
from abc import ABC, abstractmethod

# ABSTRACT CLASS -----------------------------------------------------------
class BaseModel(ABC):

    def __init__(self):
        self.feature_dict = {}
        self.covariate_range_dict = {}
        self.model_name = 'Base Model'
        self.dataframe = None
        self.input_features = []
        self.label_feature = None
        self.X = None
        self.Y = None
        self.rfe = False
        self.iterable_model_options_dict = {}

    def get_dataframe(self, dataframe):
        # Todo: implement dataframe sense check
        self.dataframe = dataframe

    def get_dataframe_code(self, dataframe_code):
        for item in dataframe_code['Variable']:
            self.feature_dict[item] = dataframe_code['Type'][dataframe_code['Variable'] == item].values[0]
    
    def get_dataframe_covariate_range(self, dataframe_covariate_range):
        for item in dataframe_covariate_range['Variable']:
            min_val = dataframe_covariate_range['Min'][dataframe_covariate_range['Variable'] == item].values[0]
            max_val =  dataframe_covariate_range['Max'][dataframe_covariate_range['Variable'] == item].values[0]
            interval = dataframe_covariate_range['Interval'][dataframe_covariate_range['Variable'] == item].values[0]
            self.covariate_range_dict[item] = [*range(min_val, max_val, interval)]

    def get_data(self, file):
        self.get_dataframe(file['Data'])
        self.get_dataframe_code(file['Data Code'])
        self.get_dataframe_covariate_range(file['Data Range'])

    def get_input_features(self, selected_features):
        self.input_features = selected_features

    def get_input_options(self):
        self.available_input_features = list()
        for key in self.dataframe.columns:
            if self.feature_dict[key] == 'con_input' or self.feature_dict[key] == 'cat_input':
                self.available_input_features.append(key)

        return self.available_input_features

    def get_label_feature(self, label_feature, censoring=False):
        if label_feature == 'Failure within 1 year [y/n]':
            self.label_feature = 'Graft failed within 1 year of transplant? '
        elif label_feature == 'Survival time [days]':
            if censoring:
                self.label_feature = 'Graft survival censored'
            else:
                self.label_feature = 'Graft survival uncensored'
        else:
            self.label_feature = label_feature

    def process_input_options(self, verbose=True):
        # Only focusing on deceased donor transplants for the purpose of this prototype
        transplant_subset = ['DBD kidney transplant', 'DCD kidney transplant']
        self.dataframe = self.dataframe[self.dataframe['Transplant type'].isin(transplant_subset)].copy()

        failure_proportion_predrop = float(sum(self.dataframe['Graft failed within 1 year of transplant? '] == 'Yes') /
                                    len(self.dataframe['Graft failed within 1 year of transplant? ']))

        if verbose:
            st.text("Uploaded dataset (deceased donor subset): {} samples".format(self.dataframe.shape[0]))
            st.text("=> i.e {:.1f} % positive for failure within 1 year".format(failure_proportion_predrop * 100))

        self.dataframe.replace('n/a', np.NaN, inplace=True)
        self.dataframe.dropna(axis=0, how='any', subset=self.input_features + [self.label_feature], inplace=True)

        if self.dataframe.shape[0] <= 0:
            st.info('❗Not enough samples to proceed analysis. Please reselect options.')
            st.stop()

        self.class_proportion = float(sum(self.dataframe['Graft failed within 1 year of transplant? '] == 'Yes') /
                            len(self.dataframe['Graft failed within 1 year of transplant? ']))

        if verbose:
            st.text("Post incomplete data removal: {} samples".format(self.dataframe.shape[0]))
            st.text("=> i.e {:.1f} % positive for failure within 1 year".format(self.class_proportion * 100))

        self.input_features_cat = []
        for feature in self.input_features:
            if self.feature_dict[feature] == 'cat_input':
                self.input_features_cat.append(feature)

        self.X = pd.get_dummies(self.dataframe[self.input_features], prefix=self.input_features_cat, columns=self.input_features_cat)
        self.Y = self.dataframe[self.label_feature]

    def get_iterable_model_options(self, *args, **kwargs):

        iterables = list()
        for option_list in args:
            iterables.append(option_list)  # used for unordered multiselect inputs; i.e. criterion

        for option in kwargs:
            select_list = np.array(self.iterable_model_options_dict[option + '_list'])
            min, max = kwargs[option]
            min_index = np.where(select_list==min)[0][0]
            max_index = np.where(select_list==max)[0][0]
            selected_list = list(select_list[min_index:max_index+1])  # select inclusively for ordered selections; i.e. c_params
            iterables.append(selected_list)

        return iterables

    def sort_feature_importance(self):
        # sort features in importance rank as outcome of recursive feature elimination
        self.sorted_features = list(range(len(self.X_train.columns)))
        i = 0
        for item in self.X_train.columns:
            index = self.selector.ranking_[i] - 1
            self.sorted_features[index] = item
            i += 1

    def train_test_split(self, test_proportion, verbose=True):
        stratify = self.Y if self.feature_dict[self.label_feature] == 'cat_output' else None
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, stratify=stratify, test_size=test_proportion)
        if verbose:
            st.text("Train size: {} samples | Positive: {}; Negative: {}".format(len(self.X_train), sum(self.Y_train == 'Yes'),
                                                                                 sum(self.Y_train == 'No')))
            st.text("Test size: {} samples | Positive: {}; Negative: {}".format(len(self.X_test), sum(self.Y_test == 'Yes'),
                                                                                sum(self.Y_test == 'No')))

    def concordance_index_score(self, estimator, X, y):
        # wrapper to implement custom sklearn scoring function using lifelines
        return concordance_index(predicted_scores=estimator(X), event_times=y)

    def plot_confusion_matrix(self):

        cm_train = confusion_matrix(y_true=self.Y_train, y_pred=self.Y_train_pred)
        cm_test = confusion_matrix(y_true=self.Y_test, y_pred=self.Y_test_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Confusion Matrix: {}".format(self.model_name))

        ConfusionMatrixDisplay(cm_train).plot(ax=ax1)
        ConfusionMatrixDisplay(cm_test).plot(ax=ax2)
        ax1.set_title('Train data')
        ax2.set_title('Test data')

        st.subheader('Confusion matrix: {}'.format(self.model_name))
        st.pyplot(fig)

    def plot_variable_importance(self):

        result = permutation_importance(self.best_classifier, self.X_train, self.Y_train, n_repeats=10, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=self.X_train.columns[sorted_idx])
        ax.set_title("Permutation Importances (train set): {}".format(self.model_name))
        fig.tight_layout()

        st.subheader('Variable importance: {}'.format(self.model_name))
        st.pyplot(fig)

    def plot_recursive_feature_elimination_cross_validation_test(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('RFE Cross Validation Outcome')

        ax1.plot(range(1, len(self.selector.cv_results_['mean_test_score']) + 1),
                 -1 * self.selector.cv_results_['mean_test_score'])
        ax1.set_ylabel('RMSE')
        ax1.set_xlabel('Number of features')

        ax2.plot(range(1, len(self.selector.cv_results_['concordance_index_test_score']) + 1),
                 self.selector.cv_results_['concordance_index_test_score'])
        ax2.set_ylabel('Concordance Index')
        ax2.set_xlabel('Number of features')

        st.pyplot(fig)
        self.fig_rfecv = fig

    def export_log_to_local(self):
        st.text('Exporting log for {}'.format(self.model_name.lower()))
        experiment_number = 0
        file_root = str(os.getcwd()) + '/Results/'

        while os.path.isdir(file_root + 'Experiment_' + str(experiment_number) + '/'):
            experiment_number += 1
        os.makedirs(file_root + 'Experiment_' + str(experiment_number) + '/')
        file_path = file_root + 'Experiment_' + str(experiment_number) + '/' + self.model_name + '.xlsx'
        st.text('Saved as {}'.format(file_path))
        self.log.to_excel(file_path)

    def export_fig_to_local(self):
        st.text('Exporting figure for {}'.format(self.model_name.lower()))
        experiment_number = 0
        file_root = str(os.getcwd()) + '/Results/'

        while os.path.isdir(file_root + 'Experiment_' + str(experiment_number) + '/Figures/'):
            experiment_number += 1
        file_path = file_root + 'Experiment_' + str(experiment_number) + '/Figures/'
        os.makedirs(file_path)

        for fig_number, fig in enumerate(self.fig_list):
            fig.savefig(file_path + 'figure_' + str(fig_number) + '.png', transparent=True)
            st.text('Saved as {}'.format(file_path))

    def create_log_download_button(self):
        log = self.log.to_csv().encode('utf-8')
        st.download_button(
            label='Download output as csv',
            data=log,
            file_name=self.model_name + '.csv',
            mime="text/csv"
        )

    def create_fig_download_button(self):

        figure_number = 0
        for fig in self.fig_list:

            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True)

            st.download_button(
                label='Download figure as png',
                data=buf.getvalue(),
                file_name=self.model_name + '_figure_' + str(figure_number) + '.png',
                mime="image/png"
            )

            figure_number += 1

    def create_zip_download_button(self):

        # create zip object
        file_name = self.model_name + '.zip'
        zipObj = zipfile.ZipFile(file_name, "w")

        # zip log
        log = self.log.to_csv().encode('utf-8')
        zipObj.write(log)

        # zip figures
        for fig in self.fig_list:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True)
            zipObj.write(buf.getvalue())

        zipObj.close()

        # create download button
        st.download_button(
            label='Download all output as zip',
            data=zipfile,
            file_name=file_name
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

    @abstractmethod
    def save_log(self):
        pass
