import http.client
from utils import *
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

def get_gptzero_response(payload: str) -> str:
    conn = http.client.HTTPSConnection("api.gptzero.me")

    headers = {
        'x-api-key': os.getenv("GPTZERO_API_KEY"),
        'Content-Type': "application/json",
        'Accept': "application/json"
    }

    conn.request("POST", "/v2/predict/text", payload, headers)

    res = conn.getresponse()
    data = res.read()

    return data.decode("utf-8")

def save_to_json(data: str, file_name: str):
    data = data.strip()

    # Parse the string into a Python dictionary
    parsed = json.loads(data)

    # Save to a file
    with open(file_name, "w") as f:
        json.dump(parsed, f, indent=2)

def read_json(file_name: str):
    with open(file_name, "r") as f:
        return json.load(f)

def get_average_generated_prob(file_name: str):
    response = read_json(file_name)
    return response['documents'][0]['average_generated_prob']

def read_applications():
    df = pd.read_csv('application.csv')
    return df

def fill_average_generated_prob(df: pd.DataFrame):
    # Transform explanations to JSON format
    json_data = transform_column_to_json(df, 'explanation', as_string=True)
    
    # Get GPTZero response for each JSON
    generated_probs = []
    for json_str in json_data:
        response = get_gptzero_response(json_str)
        save_to_json(response, "temp.json")
        prob = get_average_generated_prob("temp.json")
        generated_probs.append(prob)
    
    # Add probabilities as new column
    df['average_generated_prob'] = generated_probs
    return df

if __name__ == "__main__":
    # df = read_applications()
    # for application in df['explanation']:
    #     print(application)
    df = fill_average_generated_prob(read_applications())
    df.to_csv('application_with_average_generated_prob.csv', index=False)

    # save_to_json(processed_response, "output.json")

    # response = read_json("output.json")
    # print(response['documents'][0]['average_generated_prob'])
    # # print(get_gptzero_response(transformed_data[0]))