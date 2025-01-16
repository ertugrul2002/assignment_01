import numpy as np
import pandas as pd
import plotly.express as px
from typing import Union, List
import plotly.graph_objs as go

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

def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func: str) -> pd.DataFrame:
    """
    Group data by specified column and apply aggregation function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_by_column (str): Column to group by
        agg_func (str): Aggregation function ('sum', 'mean', or 'count')
        
    Returns:
        pd.DataFrame: Grouped and aggregated data
    """
    # Get numeric columns excluding the grouping column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != group_by_column]
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for aggregation")
    
    # Perform grouping and aggregation
    grouped = df.groupby(group_by_column)[numeric_cols].agg(agg_func)
    
    # Reset index and ensure unique column name
    if group_by_column in grouped.columns:
        # If column name conflicts, create a unique name
        new_col_name = f"{group_by_column}_group"
        counter = 1
        while new_col_name in grouped.columns:
            new_col_name = f"{group_by_column}_group_{counter}"
            counter += 1
        grouped = grouped.reset_index().rename(columns={group_by_column: new_col_name})
    else:
        grouped = grouped.reset_index()
    
    return grouped

    
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


def create_visualization(
    df: pd.DataFrame, 
    dimension_type: str, 
    hover_data: List[str],
    n_components: int = 2
) -> Union[go.Figure, None]:


    """
    Create an interactive scatter plot using Plotly, supporting both 2D and 3D visualizations.
    
    Args:
        df (pd.DataFrame): DataFrame with reduced dimensions
        dimension_type (str): 'cities' or 'parties'
        hover_data (list[str]): Columns to show in hover tooltip
        n_components (int): Number of principal components to visualize (2 or 3)
        
    Returns:
        plotly.graph_objs.Figure: Interactive scatter plot
        None: If invalid number of components specified
    """
    if n_components not in [2, 3]:
        print(f"Error: n_components must be 2 or 3, got {n_components}")
        return None
    
    # Verify required PCs exist in dataframe
    required_pcs = [f'PC{i+1}' for i in range(n_components)]
    missing_pcs = [pc for pc in required_pcs if pc not in df.columns]
    if missing_pcs:
        print(f"Error: Missing required principal components: {missing_pcs}")
        return None

    # Common parameters for both 2D and 3D plots
    plot_params = {
        'hover_data': hover_data,
        'title': f'PCA Results - {dimension_type.capitalize()}',
        'labels': {
            'PC1': 'First Principal Component',
            'PC2': 'Second Principal Component',
            'PC3': 'Third Principal Component' if n_components == 3 else None
        }
    }

    if n_components == 2:
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            **plot_params
        )
    else:  # 3D plot
        fig = px.scatter_3d(
            df,
            x='PC1',
            y='PC2',
            z='PC3',
            **plot_params
        )
        
        # Improve 3D plot layout
        fig.update_layout(
            scene = dict(
                xaxis_title='First Principal Component',
                yaxis_title='Second Principal Component',
                zaxis_title='Third Principal Component',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )

    # Common layout updates
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        title_font_size=20
    )

    return fig