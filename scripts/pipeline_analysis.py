import pandas as pd
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
PROJECT_ID = os.getenv('PROJECT_ID')
MODEL_ID = os.getenv('MODEL_ID')
URL = os.getenv('URL')

PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 300,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}

model = ModelInference(
    model_id=MODEL_ID,
    params=PARAMS,
    credentials={
        "url": URL,
        "apikey": API_KEY
    },
    project_id=PROJECT_ID,
)