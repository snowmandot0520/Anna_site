"""Tests base_model modele handling of uploaded data"""
from app.components.models import cox_ph_model

def test_load_model():
    CPH = cox_ph_model.CoxProportionalHazardsRegression()
