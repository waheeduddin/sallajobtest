import argparse, os, subprocess, sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def pip_install(package):
    subprocess.call(["pip","install", package])
    
def conda_install(package):
    subprocess.call(["conda", "install", "-c", "conda-forge", package])
# sys.executable, "-m", "pip",
def update_sklearn():
    subprocess.call(["conda", "update", "scikit-learn"])

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # preprocessing arguments
    parser.add_argument('--filename', type=str)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.filename

    # Load dataset into a pandas dataframe
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path)
    
    data.drop_duplicates(keep="first",ignore_index=True,inplace=True)
    data.dropna(inplace=True)
    target_column_name = "Class"
    X_train, y_train = data.drop(columns=target_column_name,axis="columns"), data[target_column_name]
    pre_processed_df = X_train.copy()

    scaler = MinMaxScaler()
    scaler.fit(X_train)    
    pre_processed_df[list(pre_processed_df.columns)] = scaler.transform(X_train)
    pre_processed_df[target_column_name] = y_train
    
    # Create output dirs
    scalar_output_dir = '/opt/ml/processing/output/scaler'
    fs_output_dir = '/opt/ml/processing/output/processed_data'
    os.makedirs(scalar_output_dir, exist_ok=True)
    os.makedirs(fs_output_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(scalar_output_dir,'min_max_scalar.gz'))
    fs_output_path = os.path.join(fs_output_dir, 'pre_processed_train_data.csv')    
    
    print('Saving SageMaker Feature Store training data to {}'.format(fs_output_path))
    pre_processed_df.to_csv(fs_output_path, index=False, header=True)

    print("completed!")