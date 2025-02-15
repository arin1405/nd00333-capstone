import json
import numpy as np
import os
import pickle
import joblib
import pandas as pd

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hyperdrive_breast_cancer_model.joblib')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = pd.DataFrame.from_dict(data)
        # make prediction
        mypredict = model.predict(data)
        return mypredict.tolist()
    except Exception as ex:
        error = str(ex)
        return error
