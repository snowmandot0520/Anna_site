import streamlit as st
import pandas as pd

from .App.main.models.CPH import CoxProportionalHazardsRegression

CPH = CoxProportionalHazardsRegression()

# TODO: check input data for data code, data range logic. 