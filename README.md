# Objective

We will gain hands-on experience with dimensionality reduction techniques by implementing our own functions and building a basic user interface using Streamlit. This will help us understand the fundamental concepts behind dimensionality reduction and practice developing a minimal UI to interact with the data.
We will analyze the dataset of Knesset elections in Israel, where each row represents a ballot box. The columns include: city name, ballot box number, and the number of votes given to each party. Our task is to determine how different cities cluster based on voting patterns and how different parties tend to receive votes from similar or distinct areas. We will identify and interpret clustering patterns that emerge from your analysis.


# Election Data Analysis Tool

This project implements dimensionality reduction techniques for analyzing Israeli Knesset election data, including a Streamlit-based user interface for interactive exploration.

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install numpy pandas plotly streamlit
```

## Project Structure

- `dimreduce.py`: Core implementation of data processing and analysis functions
- `app.py`: Streamlit web application
- `demo.ipynb`: Jupyter notebook demonstrating usage
- `README.md`: This file

## Usage

### Running the Streamlit App

1. Navigate to the project directory
2. Run:
```bash
streamlit run app.py
```
3. Open your web browser to the URL shown in the terminal

### Using the Python Module

```python
from dimreduce import load_data, group_and_aggregate_data, dimensionality_reduction

# Load your data
df = load_data('your_data.csv')

# Process the data
grouped_df = group_and_aggregate_data(df, 'city_name', 'sum')
reduced_df = dimensionality_reduction(grouped_df, 2, ['city_name'])
```

## Features

- Data loading from CSV and Excel files
- Flexible data grouping and aggregation
- Sparse column removal
- Custom PCA implementation
- Interactive visualizations using Plotly
- User-friendly Streamlit interface

## Implementation Details

- PCA is implemented from scratch using NumPy
- Visualizations use Plotly for interactivity
- Streamlit interface provides easy data exploration
- All functions include comprehensive docstrings