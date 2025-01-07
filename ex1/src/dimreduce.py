import numpy as np
import pandas as pd

def checkExt(filepath: str) -> int:
    if ('.csv' in filepath):
        return 0
    if ('.xlsx' in filepath):
        return 1
    return 2


def load_data(filepath: str) -> pd.DataFrame:
    # print(">>>printing filepath: ",filepath)
    # print("is csv ? :", ('.csv' in filepath))
    # print("isXlsx ? : ", ('.xlsx' in filepath))
    # print("isXls ? : ", ('.xlsx' not in filepath) and ('.csv' not in filepath))
    """
    Load data from a CSV or Excel file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if (checkExt(filepath) == 0):
        return pd.read_csv(filepath)
    elif ((checkExt(filepath) == 1) or (checkExt(filepath) == 2)):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    """
    Group and aggregate data by specified column using given aggregation function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_by_column (str): Column name to group by
        agg_func: Aggregation function to apply
        
    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df.groupby(group_by_column)[numeric_cols].agg(agg_func).reset_index()

def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove columns where the sum of values is below the specified threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (int): Minimum sum threshold for keeping a column
        
    Returns:
        pd.DataFrame: DataFrame with sparse columns removed
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    meta_cols = [col for col in df.columns if col not in numeric_cols]
    
    sums = df[numeric_cols].sum()
    cols_to_keep = sums[sums >= threshold].index
    
    return df[meta_cols + list(cols_to_keep)]

def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    Perform PCA dimensionality reduction on the data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        num_components (int): Number of principal components to retain
        meta_columns (list[str]): Columns to exclude from PCA
        
    Returns:
        pd.DataFrame: DataFrame with reduced dimensions plus metadata
    """
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