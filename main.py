import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly as py
import openpyxl


def load_data(filepath: str) -> pd.DataFrame:

    if filepath.endswith('.csv'):
        data1 = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        data1 = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

    return data1




#def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
#def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
#def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
#def visualize_data(df: pd.DataFrame, num_components: int) -> None:
#def create_streamlit_ui():





filepath = "knesset_25.xlsx"
data = load_data(filepath=filepath)

print(data.head())