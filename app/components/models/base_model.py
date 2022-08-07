'''Implements BaseModel which is an abstract class to be inherited by all model implementations.'''

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
    '''Implements BaseModel which is an abstract class to be inherited by all model implementations.'''
    def __init__(self):
        self.feature_dict = {}
        self.covariate_range_dict = {}
        self.model_name = 'Base Model'
        self.dataframe = None
        self.input_features = []
        self.label_feature = None
        self.x = None
        self.y = None
        self.rfe = False
        self.iterable_model_options_dict = {}

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

    def get_dataframe(self, dataframe):
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
        self.available_input_features = []
        for key in self.dataframe.columns:
            if self.feature_dict[key] == 'con_input' or self.feature_dict[key] == 'cat_input':
                self.available_input_features.append(key)
        return self.available_input_features

    def calculate_survival(self):
        self.survival = self.dataframe['Date of failure (kidney)'] - self.dataframe['Date of Transplant']
        self.survival.where(self.survival > 0, 0, inplace=True)
        return self.survival

    def censor_survival(self, df_to_censor, censor_duration_days):
        self.calculate_survival()
        self.censored_survival = df_to_censor.where(
            df_to_censor < censor_duration_days,
            censor_duration_days
        )
        return self.censored_survival

    def create_boolean_survival_status(self, cutoff_duration_days):
        self.calculate_survival()
        self.boolean_survival_status = self.survival.where(
            self.survival < cutoff_duration_days,
            True
        )
        self.boolean_survival_status.where(
            self.survival >= cutoff_duration_days,
            False, inplace=True
        )
        return self.boolean_survival_status

    def create_label_feature(self, label_feature, censoring:bool, duration:int):
        if label_feature == 'Failure within given duration [y/n]':
            self.label_feature = f'Failure within {duration} days [y/n]'
            label = self.create_boolean_survival_status(cutoff_duration_days=duration)
        elif label_feature == 'Survival time [days]':
            if censoring:
                self.label_feature = f'Survival time censored to {duration} [days]'
                label = self.censor_survival(
                    df_to_censor=self.calculate_survival(),
                    censor_duration_days=duration
                    )
            else:
                self.label_feature = 'Survival time uncensored [days]'
                label = self.calculate_survival()
        else:
            raise ValueError(f'defined label: {label_feature} is not defined')

        self.label = pd.Series(data=label, name=self.label_feature)
        self.dataframe = pd.concat([self.dataframe, self.label], axis=1)

    def process_input_options(self):
        # Only focusing on deceased donor transplants for the purpose of this prototype
        transplant_subset = ['DBD kidney transplant', 'DCD kidney transplant']
        self.dataframe = self.dataframe[self.dataframe['Transplant type'].isin(transplant_subset)].copy()
        self.dataframe.replace('n/a', np.NaN, inplace=True)
        self.dataframe.dropna(axis=0, how='any', subset=self.input_features + [self.label_feature], inplace=True)

        if self.dataframe.shape[0] <= 0:
            st.info('❗Not enough samples to proceed analysis. Please reselect options.')
            st.stop()

        self.input_features_cat = []
        for feature in self.input_features:
            if self.feature_dict[feature] == 'cat_input':
                self.input_features_cat.append(feature)

        self.x = pd.get_dummies(
            self.dataframe[self.input_features],
            prefix=self.input_features_cat,
            columns=self.input_features_cat
            )
        self.y = self.dataframe[self.label_feature]

    def get_iterable_model_options(self, *args, **kwargs):
        iterables = []
        for option_list in args:
            # used for unordered multiselect inputs; i.e. criterion
            iterables.append(option_list)
        for option, value in kwargs.items():
            select_list = np.array(self.iterable_model_options_dict[option + '_list'])
            min_val, max_val = value
            min_index = np.where(select_list==min_val)[0][0]
            max_index = np.where(select_list==max_val)[0][0]
            # select inclusively for ordered selections; i.e. c_params
            selected_list = list(select_list[min_index:max_index+1])
            iterables.append(selected_list)
        return iterables

    def sort_feature_importance(self):
        # sort features in importance rank as outcome of recursive feature elimination
        self.sorted_features = list(range(len(self.x_train.columns)))
        i = 0
        for item in self.x_train.columns:
            index = self.selector.ranking_[i] - 1
            self.sorted_features[index] = item
            i += 1

    def train_test_split(self, test_proportion, verbose):
        stratify = self.y if self.label_feature[-5:] == '[y/n]' else None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                                                                    self.x,
                                                                    self.y,
                                                                    stratify=stratify,
                                                                    test_size=test_proportion
                                                                    )
        if verbose:
            st.text(
                f'Train size: {len(self.x_train)} samples | '
                f'Positive: {sum(self.y_train == "Yes")}; '
                f'Negative: {sum(self.y_train == "No")}'
            )
            st.text(
                f'Test size: {len(self.x_test)} samples | '
                f'Positive: {sum(self.y_test == "Yes")}; '
                f'Negative: {sum(self.y_test == "No")}'
            )

    def concordance_index_score(self, estimator, x, y):
        # wrapper to implement custom sklearn scoring function using lifelines
        return concordance_index(predicted_scores=estimator(x), event_times=y)

    def plot_confusion_matrix(self):
        cm_train = confusion_matrix(y_true=self.y_train, y_pred=self.y_train_pred)
        cm_test = confusion_matrix(y_true=self.y_test, y_pred=self.y_test_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Confusion Matrix: {self.model_name}')
        ConfusionMatrixDisplay(cm_train).plot(ax=ax1)
        ConfusionMatrixDisplay(cm_test).plot(ax=ax2)
        ax1.set_title('Train data')
        ax2.set_title('Test data')
        st.subheader(f'Confusion matrix: {self.model_name}')
        st.pyplot(fig)

    def plot_variable_importance(self):
        result = permutation_importance(self.best_classifier, self.x_train, self.y_train, n_repeats=10, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=self.x_train.columns[sorted_idx])
        ax.set_title(f'Permutation Importances (train set): {self.model_name}')
        fig.tight_layout()
        st.subheader(f'Variable importance: {self.model_name}')
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
        st.text(f'Exporting log for {self.model_name.lower()}')
        experiment_number = 0
        file_root = str(os.getcwd()) + '/Results/'

        while os.path.isdir(file_root + 'Experiment_' + str(experiment_number) + '/'):
            experiment_number += 1
        os.makedirs(file_root + 'Experiment_' + str(experiment_number) + '/')
        file_path = (file_root + 'Experiment_' + str(experiment_number) + '/'
                    + self.model_name + '.xlsx')
        st.text(f'Saved as {file_path}')
        self.log.to_excel(file_path)

    def export_fig_to_local(self):
        st.text(f'Exporting figure for {self.model_name.lower()}')
        experiment_number = 0
        file_root = str(os.getcwd()) + '/Results/'

        while os.path.isdir(file_root + 'Experiment_' + str(experiment_number) + '/Figures/'):
            experiment_number += 1
        file_path = file_root + 'Experiment_' + str(experiment_number) + '/Figures/'
        os.makedirs(file_path)

        for fig_number, fig in enumerate(self.fig_list):
            fig.savefig(file_path + 'figure_' + str(fig_number) + '.png', transparent=True)
            st.text(f'Saved as {file_path}')

    def create_log_download_button(self):
        log = self.log.to_csv().encode('utf-8')
        st.download_button(
            label='Download output as csv',
            data=log,
            file_name=self.model_name + '.csv',
            mime='text/csv'
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
                mime='image/png'
            )
            figure_number += 1

    def create_zip_download_button(self):
        file_name = self.model_name + '.zip'
        zip_obj = zipfile.ZipFile(file_name, 'w')
        log = self.log.to_csv().encode('utf-8')
        zip_obj.write(log)
        for fig in self.fig_list:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True)
            zip_obj.write(buf.getvalue())
        zip_obj.close()
        st.download_button(
            label='Download all output as zip',
            data=zipfile,
            file_name=file_name
        )

