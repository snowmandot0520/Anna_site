"""Tests base_model modele handling of uploaded data"""
from app.components.models import cox_ph_model

def test_load_model():
    model = cox_ph_model.CoxProportionalHazardsRegression()

def test_load_data():
    pass

def test_data_format():
    pass
