# LOAD DEPENDENCY ----------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from App.main.models.base_model import BaseModel
from main.utils.RFECV import RFECV
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_validate


# DEFINE MODEL -------------------------------------------------------------
class GaussianProcess(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = "Gaussian Process"
        self.iterable_model_options_dict = {}

    def build_estimator(self):
        pass

    def train(self, verbose=False):
        pass

    def evaluate(self, verbose=False):
        pass

    def visualize(self):
        pass

    def save_log(self):
        cache = {}
        self.log = pd.DataFrame(data=cache)

    def save_fig(self):
        self.fig_list = []
