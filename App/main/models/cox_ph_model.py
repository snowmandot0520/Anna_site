# LOAD DEPENDENCY ----------------------------------------------------------
from tabnanny import verbose
from typing import List
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from App.main.models.base_model import BaseModel
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


# CLASS  OBJECT -----------------------------------------------------------
class CoxProportionalHazardsRegression(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = 'Cox Proportional Hazards'
        self.iterable_model_options_dict = {}

    def create_event_status(self, duration_days: int):
        self.x['Event_observed'] = np.zeros(self.y.size)
        for idx, duration in self.y.iteritems():
            if duration < duration_days:
                self.x.at[idx, 'Event_observed'] = 1

    def combine_outcome_data(self):
        # combine outcome into same dataframe for CoxPH
        self.train_dataframe = pd.concat([self.x_train, self.y_train], axis=1)
        self.test_dataframe = pd.concat([self.x_test, self.y_test], axis=1)

    def build_estimator(self):
        self.estimator = CoxPHFitter()

    def train(self):
        self.estimator.fit(self.train_dataframe, duration_col=self.y.name, event_col='Event_observed')

    def evaluate(self, verbose=verbose):
        # Train
        train_prediction = self.estimator.predict_expectation(self.train_dataframe)
        self.train_concordance_index = concordance_index(event_times=self.train_dataframe['Graft survival censored'], predicted_scores=train_prediction,
                                                event_observed=self.train_dataframe['Event_observed'])
        # Test
        test_prediction = self.estimator.predict_expectation(self.test_dataframe)
        self.test_concordance_index = concordance_index(event_times=self.test_dataframe['Graft survival censored'], predicted_scores=test_prediction,
                                                event_observed=self.test_dataframe['Event_observed'])
        st.write('Train concordance index: {:.3f}'.format(self.train_concordance_index))
        st.write('Test concordance index: {:.3f}'.format(self.test_concordance_index))
    
        if verbose:
            st.write(self.estimator.summary)

    def plot_coefficients(self):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Coefficient plot')
            self.estimator.plot(ax=ax)
            st.pyplot(fig)

    def plot_partial_effects(self, selected_covariates):

        if len(selected_covariates) != 0:

            fig_multi= plt.figure(figsize=(8, 4*len(selected_covariates)))
            gs = gridspec.GridSpec(len(selected_covariates), 1)

            for i, covariate in enumerate(selected_covariates):
                ax = fig_multi.add_subplot(gs[i])
                ax.set_title(covariate)
                self.estimator.plot_partial_effects_on_outcome(covariates=covariate, values=self.covariate_range_dict[covariate], ax=ax)

            gs.tight_layout(fig_multi)
            st.pyplot(fig_multi)

    def visualize(self, selected_covariates):
        with st.expander('Plot partial effects on outcomes'):
            self.plot_partial_effects(selected_covariates)
        with st.expander('Plot Coefficients'):
            self.plot_coefficients()

    def save_log(self):
        cache = {'model': self.model_name, 'input_features': self.input_features, 'label_feature': self.label_feature,
            'train_concordance_index': self.train_concordance_index, 'test_concordance_index': self.test_concordance_index
        }
        self.log = pd.DataFrame(data=cache)

    def save_fig(self):
        self.fig_list = []