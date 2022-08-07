# LOAD DEPENDENCY ----------------------------------------------------------
import pandas as pd

from app.components.models.base_model import BaseModel
from sklearn.metrics import *


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
