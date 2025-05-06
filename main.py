import http.client
from utils import *
import os
from dotenv import load_dotenv
import json
import pandas as pd
from tqdm import tqdm
load_dotenv(override=True)

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

def read_applications():
    df = pd.read_csv('application.csv')
    return df

def fill_average_generated_prob(df: pd.DataFrame):
    # Transform explanations to JSON format
    json_data = transform_column_to_json(df, 'explanation', as_string=True)
    
    # Get GPTZero response for each JSON
    generated_probs = []
    for json_str in tqdm(json_data):
        response = get_gptzero_response(json_str)
        prob = response['documents'][0]['average_generated_prob']
        generated_probs.append(prob)
    # Add probabilities as new column
    df['average_generated_prob'] = generated_probs
    return df

if __name__ == "__main__":

    print(os.getenv("GPTZERO_API_KEY"))
    df = fill_average_generated_prob(read_applications())
    df.to_csv('application_with_average_generated_prob.csv', index=False)