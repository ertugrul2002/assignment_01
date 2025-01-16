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
    
    num_components = st.slider("Number of principal components", min_value=2, max_value=5, value=2)
    visualization_list = []
    pca_subheader = "PCA Visualization"
    if num_components == 2:
        visualization_list = ['2D']
    elif num_components == 3:
        visualization_list = ['2D', '3D']
    else:
        pca_subheader = "Can Not Visualize more than 3D Graphs"

        visualization_list = []
  

    # Visualization options
    viz_type = st.radio("Select visualization type", visualization_list)
    if viz_type == '3D':
        num_components = 3
    elif viz_type == '2D':
        num_components = 2
    else:
        num_components = num_components
    
    if analysis_type == 'Parties':
        # Transpose for party analysis
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        analysis_df = filtered_df[numeric_cols].T
        meta_columns = []
        hover_data = []  # No additional hover data for parties
    else:
        analysis_df = filtered_df
        meta_columns = [group_col]
        hover_data = [group_col]  # Include group column in hover data
    
    # Perform dimensionality reduction
    reduced_df = dimensionality_reduction(analysis_df, num_components, meta_columns)
    
    # Create visualization
    st.subheader(pca_subheader)
    
    # Add visualization settings
    st.sidebar.subheader("Visualization Settings")
    point_size = st.sidebar.slider("Point Size", min_value=5, max_value=30, value=10)
    
    # Create the visualization with the specified number of components
    fig = create_visualization(
        reduced_df,
        analysis_type.lower(),
        hover_data,
        n_components=num_components
    )
    
    # Update marker size
    if fig is not None:
        if viz_type == '2D':
            fig.update_traces(marker=dict(size=point_size))
        else:
            fig.update_traces(marker=dict(size=point_size))
    
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of visualization
        if viz_type == '2D':
            st.info("""
                The 2D plot shows the first two principal components, which capture the main patterns in the data.
                Points that are closer together indicate similar voting patterns.
            """)
        else:
            st.info("""
                The 3D plot shows the first three principal components, providing an additional dimension
                to visualize patterns in the data. You can rotate and zoom the plot to explore different angles.
                Use your mouse to interact with the visualization.
            """)
    
    # Display reduced data
    st.subheader("Reduced Data")
    st.dataframe(reduced_df)
    
    # Add download button for reduced data
    csv = reduced_df.to_csv(index=True)
    st.download_button(
        label="Download reduced data as CSV",
        data=csv,
        file_name="reduced_data.csv",
        mime="text/csv"
    )