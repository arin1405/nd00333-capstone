import json
import logging
import os
import numpy as np
import pandas as pd
import joblib
from azureml.core import Workspace
from azureml.core.model import Model

def init():
    global model
    ws = Workspace.from_config()
    print(Model.get_model_path("best_model", version=2, _workspace=ws))
    model_path = Model.get_model_path("best_model", version=2, _workspace=ws) 
    model = joblib.load(model_path)

def run(input_data):
    try:
        predictions = model.predict(pd.DataFrame(json.loads(input_data)['data']))
        return json.dumps({"result": predictions.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})