import pandas as pd
import json
from typing import List, Dict, Any, Union

def read_applications(file_path: str = 'application.csv') -> pd.DataFrame:
    """
    Read the applications CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file. Defaults to 'application.csv'
    
    Returns:
        pd.DataFrame: DataFrame containing the applications data
    """
    return pd.read_csv(file_path)

def add_column(df: pd.DataFrame, column_name: str, data: List[Any]) -> pd.DataFrame:
    """
    Add a new column to the applications DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the new column
        data (List[Any]): List of values for the new column
    
    Returns:
        pd.DataFrame: DataFrame with the new column added
    
    Raises:
        ValueError: If the length of data doesn't match the DataFrame length
    """
    if len(data) != len(df):
        raise ValueError(f"Length of data ({len(data)}) must match DataFrame length ({len(df)})")
    
    df[column_name] = data
    return df

def save_applications(df: pd.DataFrame, file_path: str = 'application.csv') -> None:
    """
    Save the DataFrame back to CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (str): Path where to save the CSV file. Defaults to 'application.csv'
    """
    df.to_csv(file_path, index=False)

def transform_column_to_json(df: pd.DataFrame, column_name: str, num_rows: int = None, as_string: bool = False) -> Union[List[Dict[str, Any]], List[str]]:
    """
    Transform a column into a list of JSON objects with the specified format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column to transform
        num_rows (int, optional): Number of rows to select. If None, selects all rows.
        as_string (bool, optional): If True, returns JSON as formatted strings. Defaults to False.
    
    Returns:
        Union[List[Dict[str, Any]], List[str]]: If as_string is False, returns list of dictionaries.
        If as_string is True, returns list of formatted JSON strings.
    """
    # Select the specified number of rows if num_rows is provided
    selected_df = df.head(num_rows) if num_rows is not None else df
    
    json_objects = [
        {
            "document": str(text),
            "multilingual": False
        }
        for text in selected_df[column_name]
    ]
    
    if as_string:
        return [json.dumps(obj, indent=2) for obj in json_objects]
    return json_objects

if __name__ == "__main__":
    df = read_applications()
    for application in df['explanation']:
        print(application)
        print('--------------------------------')