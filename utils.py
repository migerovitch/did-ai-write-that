import pandas as pd
import json
from typing import List, Dict, Any, Union
import http.client
import os
from tqdm import tqdm

def get_gptzero_response(payload: str) -> dict:
    conn = http.client.HTTPSConnection("api.gptzero.me")

    headers = {
        'x-api-key': os.getenv("GPTZERO_API_KEY"),
        'Content-Type': "application/json",
        'Accept': "application/json"
    }

    conn.request("POST", "/v2/predict/text", payload, headers)

    res = conn.getresponse()
    data = res.read()
    response_str = data.decode("utf-8")
    return json.loads(response_str)

def fill_average_generated_prob(df: pd.DataFrame, field_name: str):
    # Transform explanations to JSON format
    json_data = transform_column_to_json(df, field_name, as_string=True)
    
    # Get GPTZero response for each JSON
    generated_probs = []
    for json_str in tqdm(json_data):
        response = get_gptzero_response(json_str)
        prob = response['documents'][0]['average_generated_prob']
        generated_probs.append(prob)
    # Add probabilities as new column
    df['average_generated_prob'] = generated_probs
    return df

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