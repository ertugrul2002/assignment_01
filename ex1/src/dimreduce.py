import numpy as np
import pandas as pd
import plotly
import plotly.express as px

#1st function (loading data)
def load_data(filepath: str) -> pd.DataFrame:

    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")



#2nd function (grouping,aggregating deleting the 'ballot_code' column and and putting new indexes instead of 'city_name')


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    return df.drop(columns='ballot_code').groupby(group_by_column).agg(agg_func).reset_index()



 #3rd function: deleting numeric columns from the df whose the sum of value under the specified.

def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    meta_cols = [col for col in df.columns if col not in numeric_cols]
    
    sums = df[numeric_cols].sum()
    cols_to_keep = sums[sums >= threshold].index
    
    return df[meta_cols + list(cols_to_keep)]



#4th function: dimensionality reduction with PCA

def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:

    # separate metadata and features
    meta_data = df[meta_columns]
    features = df.drop(columns=meta_columns)
    
    # standardize the features
    features_standardized = (features - features.mean()) / features.std()
    
    # calculate covariance matrix
    covariance_matrix = np.cov(features_standardized.T)
    
    # calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # select top k eigenvectors
    selected_vectors = eigenvectors[:, :num_components]

    # project data in a new space

    transformed_data = np.dot(features_standardized, selected_vectors)


    # create DataFrame with reduced dimensions

    reduced_df = pd.DataFrame(
        transformed_data,
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=df.index
    )
    
    # Add back metadata columns
    return pd.concat([meta_data, reduced_df], axis=1)


#interactive visualization using  plotly of the df after dimensional reduction:

def create_visualization(df: pd.DataFrame, dimension_type: str, hover_data: list[str]) -> 'plotly.graph_objs.Figure':


   # visualizing using scattering the data
    fig = px.scatter(
        df,
        x='PC1' ,
        y='PC2',
        hover_data = hover_data,
        title = f'PCA Results - { dimension_type.capitalize() }',
        labels = { 'PC1' : 'First Principal Component' , 'PC2' : 'Second Principal Component' }
    )
    return fig