import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.core.run import Run
from azureml.core import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

try:
    run  = Run.get_context()
    workspace = run.experiment.workspace
except:
    workspace = Workspace.from_config()

dataset = Dataset.get_by_name(workspace, name='breast_cancer_data')
ds = dataset.to_pandas_dataframe()

def process_data(data):
    x_df = data.dropna()
    y_df = x_df.pop("diagnosis")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    x, y = process_data(ds)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    val_accuracy = model.score(x_test, y_test)
    #run.log("Accuracy: ", np.float(accuracy))
    
    run_logger = Run.get_context()
    run_logger.log("accuracy", float(val_accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrive_breast_cancer_model.joblib')

if __name__ == '__main__':
    main()