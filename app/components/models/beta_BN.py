# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st
import bnlearn as bn
import matplotlib.pyplot as plt
import networkx as nx

from app.components.models.base_model import BaseModel


# DEFINE MODEL -------------------------------------------------------------
class BayesianNetwork(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = "Bayesian Network"

    def train(self):
        with st.spinner('learning structure...'):
            self.network = bn.structure_learning.fit(self.X)
        st.success('Done!')

    def visualize(self):
        with st.spinner('creating image...'):
            G = bn.plot(self.network)
            st.write(G)
            st.write(type(G))

            print('section')
            nx.draw(G)
            plt.show()

        st.success('Done!')

    def evaluate(self):
        pass

    def save_log(self):
        pass
