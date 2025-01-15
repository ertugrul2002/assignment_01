import streamlit as st
import pandas as pd
import numpy as np
from dimreduce import (
    load_data,
    group_and_aggregate_data,
    remove_sparse_columns,
    dimensionality_reduction,
    create_visualization
)

st.title("Election Data Analysis Tool")

# File upload
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    # Display raw data
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    # Grouping options
    st.subheader("Data Aggregation")
    group_col = st.selectbox("Select column to group by", df.columns)
    agg_func = st.selectbox("Select aggregation function", ['sum', 'mean', 'count'])
    
    # Group data
    grouped_df = group_and_aggregate_data(df, group_col, agg_func)
    
    # Threshold for sparse columns
    threshold = st.number_input("Minimum votes threshold", min_value=0, value=1000)
    filtered_df = remove_sparse_columns(grouped_df, threshold)
    
    # Dimensionality reduction options
    st.subheader("Dimensionality Reduction")
    analysis_type = st.radio("Select analysis type", ['Cities', 'Parties'])
    
    if analysis_type == 'Parties':
        # Transpose for party analysis
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        analysis_df = filtered_df[numeric_cols].T
        meta_columns = []
    else:
        analysis_df = filtered_df
        meta_columns = [group_col]
    
    num_components = st.slider("Number of principal components", min_value=2, max_value=5, value=2)
    
    # Perform dimensionality reduction
    reduced_df = dimensionality_reduction(analysis_df, num_components, meta_columns)
    
    # Create visualization
    st.subheader("PCA Visualization")
    fig = create_visualization(
        reduced_df,
        analysis_type.lower(),
        [group_col] if analysis_type == 'Cities' else []
    )
    st.plotly_chart(fig)
    
    # Display reduced data
    st.subheader("Reduced Data")
    st.dataframe(reduced_df)