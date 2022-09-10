'''Implements BaseModel which is an abstract class to be inherited by all model implementations.'''

# LOAD DEPENDENCY ----------------------------------------------------------
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from lifelines import KaplanMeierFitter, NelsonAalenFitter
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
        self.verbose = False
        self.boolean_survival_status = None
        self.log = []
        self.fig_list = []
        self.experiment_fig_list = []
        self.stored_best_performance = 0

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

    def create_dataframe_subset_dict(self, dataframe_subset_analysis):
        subset_dict = {}
        for row in dataframe_subset_analysis.itertuples():
            var_name = row.Variable
            var_type = row.Type
            if var_type == 'con_input':
                subset_dict[var_name] = {'type':var_type, 'value':[row.Min, row.Max]}
            elif var_type == 'cat_input':
                categories_list = row.Categories.split(',')
                subset_dict[var_name] = {'type':var_type, 'value':categories_list}
        self.selected_subset_dict = subset_dict

    def get_subset_options_dict(self):
        if self.verbose:
            print(f'Running {self.model_name} on subset of following features {self.selected_subset_dict}')
        return self.selected_subset_dict

    def get_data(self, file):
        self.get_dataframe(file['Data'])
        self.get_dataframe_code(file['Data Code'])
        self.get_dataframe_covariate_range(file['Data Range'])
        self.create_dataframe_subset_dict(file['Subset Analysis'])

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

    def create_binary_survival_status(self, cutoff_duration_days):
        self.calculate_survival()
        self.boolean_survival_status = self.survival.where(
            self.survival < cutoff_duration_days,
            1
        )
        self.boolean_survival_status.where(
            self.survival >= cutoff_duration_days,
            0, inplace=True
        )
        return self.boolean_survival_status

    def create_label_feature(self, label_feature, censoring:bool, duration:int):
        if label_feature == 'Failure within given duration [y/n]':
            self.label_feature = f'Failure within {duration} days [y/n]'
            label = self.create_binary_survival_status(cutoff_duration_days=duration)
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

    def create_event_status(self, duration_days: int):
        # Required for Cox proportional hazards model and plot_univariate_survival_curve()
        self.x['Event_observed'] = np.zeros(self.y.size)
        for idx, duration in self.y.iteritems():
            if duration < duration_days:
                self.x.at[idx, 'Event_observed'] = 1

    def remove_event_status(self):
        self.x.drop(columns=['Event_observed'])

    def create_dataframe_subset(self, subset_feature_dict):
        '''
        Summary:
            Creates a subset of the provided dataframe given the following dictionary.
            Currently no interactive widget provided to access this function from the web app.

        Args:
            subset_feature_dict = {feature_name: [selected_keys]}

        Example:
            subset_feature_dict = {'Transplant type': ['DBD kidney transplant', 'DCD kidney transplant']}
            self.create_dataframe_subset(subset_dict)
        '''
        modified_dataframe = self.dataframe
        for feature_name, nested_dict in subset_feature_dict.items():
            if nested_dict['type'] == 'cat_input':
                selected_key_list = nested_dict['value']
                modified_dataframe = modified_dataframe[modified_dataframe[feature_name].isin(selected_key_list)].copy()
            if nested_dict['type'] == 'con_input':
                min_val, max_val = nested_dict['value'][0], nested_dict['value'][1]
                modified_dataframe = modified_dataframe[modified_dataframe[feature_name].between(min_val, max_val, inclusive='both')].copy()
        self.dataframe = modified_dataframe

    def process_input_options(self):
        '''
        Summary:
            Remove NaN, check for empty uploads, create one-hot-encoding for categorical input

        Returns:
            self.x, self.y
        '''
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
        return self.x, self.y

    def get_iterable_model_options(self, **kwargs):
        '''
        Summary: create iterable list of hyperparameters to train over

        Note:
            requires all kwarg inputs to be declared in the __init__ function of each model
            unordered: unordered multiselect inputs; i.e. criterion
            ordered: ordered selections to include as a range; i.e. c_params

        Returns:
            Dict -> {option_name: iterable_selected_list}
        '''
        iterables = {}
        for option, value in kwargs.items():
            # check for declaration
            assert option in self.option_widget_type_dict, 'parameter type undefined in model class'

            if self.option_widget_type_dict[option] == 'unordered':
                iterables[option] = value
            elif self.option_widget_type_dict[option] == 'ordered':
                select_list = np.array(self.iterable_model_options_dict[option + '_list'])
                min_val, max_val = value
                min_index = np.where(select_list==min_val)[0][0]
                max_index = np.where(select_list==max_val)[0][0]
                selected_list = list(select_list[min_index:max_index+1])
                iterables[option] = selected_list
        return iterables

    def sort_feature_importance(self):
        # sort features in importance rank as outcome of recursive feature elimination
        self.sorted_features = list(range(len(self.x_train.columns)))
        i = 0
        for item in self.x_train.columns:
            index = self.selector.ranking_[i] - 1
            self.sorted_features[index] = item
            i += 1

    def train_test_split(self, test_proportion):
        stratify = self.y if self.label_feature[-5:] == '[y/n]' else None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                                                                    self.x,
                                                                    self.y,
                                                                    stratify=stratify,
                                                                    test_size=test_proportion
                                                                    )
        if self.boolean_survival_status is None:
            st.text(f'Train size: {len(self.x_train)} samples')
            st.text(f'Test size: {len(self.x_test)} samples')
        else:
            st.text(
                f'Train size: {len(self.x_train)} samples | '
                f'Positive: {sum(self.y_train == 1)}; '
                f'Negative: {sum(self.y_train == 0)}'
            )
            st.text(
                f'Test size: {len(self.x_test)} samples | '
                f'Positive: {sum(self.y_test == 1)}; '
                f'Negative: {sum(self.y_test == 0)}'
            )

    def concordance_index_score(self, estimator, x, y):
        # wrapper to implement custom sklearn scoring function using lifelines
        return concordance_index(predicted_scores=estimator(x), event_times=y)

    def store_best_estimator(self, metric):
        # store best_estimator in experiment mode
        # TODO: currently only works for metrics where higher value is better
        assert len(self.log) > 0, 'function must be called after _model.save_log()'
        assert f'test_{metric}' in self.log[-1], f'metric [{metric}] must match the string defined in the evaluate function'
        best_performance = self.log[-1][f'test_{metric}']
        if best_performance > self.stored_best_performance:
            self.stored_best_performance = best_performance
            self.stored_best_estimator = self.best_estimator
            self.stored_best_estimator_params = self.log[-1]

    def store_estimator_plots(self):
        '''
        Summary:
            Method called in experiment mode to save plots of models
            trained during iterative parameter runs.

        Returns:
            List of dictionary items with length N, where N is the
            total number of trained parameter variations
            [
                {
                    'model_ver': self.model_name + '_experiment_number'
                    'model_plots': {
                        'confusion_matrix': confusion matrix
                        'variable_importance': variable importance plot
                    }
                }
            ]
        '''
        assert len(self.log) > 0, 'function must be called after _model.save_log()'
        if self.label_feature[-5:] == '[y/n]':
            self.plot_confusion_matrix()
            self.plot_variable_importance()
            cm = self.confusion_matrix_plot
            vi = self.variable_importance_plot
        elif self.label_feature[-5:] == '[days]':
            print('To be implemented: no visualizations implemented for regression task')
        self.experiment_fig_list.append(
            {
                'model_ver':self.model_name + f'_{len(self.log)}',
                'plots':{
                    'confusion_matrix':cm,
                    'variable_importance':vi
                }
            }
        )
        return self.experiment_fig_list

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
        self.confusion_matrix_plot = fig

    def plot_variable_importance(self):
        result = permutation_importance(self.best_estimator, self.x_train, self.y_train, n_repeats=10, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=self.x_train.columns[sorted_idx])
        ax.set_title(f'Permutation Importances (train set): {self.model_name}')
        fig.tight_layout()
        st.subheader(f'Variable importance: {self.model_name}')
        st.pyplot(fig)
        self.variable_importance_plot = fig

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

    def plot_univariate_survival_curve(self):
        fig_uni, (ax1_uni, ax2_uni) = plt.subplots(1, 2, figsize=(12, 4))

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=np.asarray(self.y),
            event_observed=np.asarray(self.x['Event_observed']),
            label='Kaplan Meier Estimate'
            )
        kmf.plot(ax=ax1_uni)
        ax1_uni.set_title('Kaplan Meier survival estimate')
        ax1_uni.set_xlabel('Time [days]')
        ax1_uni.set_ylabel('Survival probability')

        naf = NelsonAalenFitter()
        naf.fit(
            durations=np.asarray(self.y),
            event_observed=np.asarray(self.x['Event_observed']),
            label='Nelson Aalen Estimate'
            )
        naf.plot_cumulative_hazard(ax=ax2_uni)
        ax2_uni.set_title('Nelson Aalen hazard estimate')
        ax2_uni.set_xlabel('Time [days]')
        ax2_uni.set_ylabel('Cumulative hazard')

        st.pyplot(fig_uni)

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
        log = pd.DataFrame(data=self.log).to_csv().encode('utf-8')
        st.download_button(
            label=f'{self.model_name} :Download output as csv',
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
        assert len(self.log) > 0, 'Error: No log to save or export'
        pd.DataFrame(data=self.log).to_csv('experiment_log.csv')
        file_name = f'local/{self.model_name}.zip'

        with zipfile.ZipFile(file_name, 'w') as zip_obj:
            zip_obj.write('local/experiment_log.csv')
            for dict_item in self.experiment_fig_list:
                name = dict_item['model_ver']
                cm, vi = dict_item['plots']['confusion_matrix'], dict_item['plots']['variable_importance']
                cm.savefig(f'local/{name}_cm.png', format='png', transparent=True)
                vi.savefig(f'local/{name}_vi.png', format='png', transparent=True)
                zip_obj.write(f'local/{name}_cm.png')
                zip_obj.write(f'local/{name}_vi.png')
                os.remove(f'local/{name}_cm.png')
                os.remove(f'local/{name}_vi.png')
            zip_obj.close()

        with open(file_name, mode="rb") as zip_file:
            st.download_button(
                label='Download all output as zip',
                data=zip_file.read(),
                file_name=self.model_name+'.zip'
            )
