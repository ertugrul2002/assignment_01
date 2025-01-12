import numpy as np
import pandas as pd

#1st function (loading data)
def load_data(filepath: str) -> pd.DataFrame:

    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")



#2nd function (grouping,aggregating and deleting the 'ballot_code' column)

def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func)-> pd.DataFrame:

 return df.drop(columns='ballot_code').groupby(group_by_column).agg(agg_func)




def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    meta_cols = [col for col in df.columns if col not in numeric_cols]
    
    sums = df[numeric_cols].sum()
    cols_to_keep = sums[sums >= threshold].index
    
    return df[meta_cols + list(cols_to_keep)]





def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:

    # Separate metadata and features
    meta_data = df[meta_columns]
    features = df.drop(columns=meta_columns)
    
    # Standardize the features
    features_standardized = (features - features.mean()) / features.std()
    
    # Calculate covariance matrix
    covariance_matrix = np.cov(features_standardized.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k eigenvectors
    selected_vectors = eigenvectors[:, :num_components]
    
    # Project data onto new space
    transformed_data = np.dot(features_standardized, selected_vectors)
    
    # Create DataFrame with reduced dimensions
    reduced_df = pd.DataFrame(
        transformed_data,
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=df.index
    )
    
    # Add back metadata columns
    return pd.concat([meta_data, reduced_df], axis=1)

def create_visualization(df: pd.DataFrame, dimension_type: str, hover_data: list[str]) -> 'plotly.graph_objs.Figure':
    """
    Create an interactive scatter plot using Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame with reduced dimensions
        dimension_type (str): 'cities' or 'parties'
        hover_data (list[str]): Columns to show in hover tooltip
        
    Returns:
        plotly.graph_objs.Figure: Interactive scatter plot
    """
    import plotly.express as px
    
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        hover_data=hover_data,
        title=f'PCA Results - {dimension_type.capitalize()}',
        labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'}
    )
    
    return fig