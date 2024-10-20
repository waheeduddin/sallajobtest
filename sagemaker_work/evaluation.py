import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost
import argparse

import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # preprocessing arguments
    parser.add_argument('--testFilename', type=str)
    parser.add_argument('--scalerFilename', type=str)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.testFilename
    scaler_file_name = args.scalerFilename
    
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path)
    
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    model = pickle.load(open("xgboost-model", "rb"))
    
    
    
    min_max_scalar_path = f"/opt/ml/processing/input/min_max_scalar.gz"
    scaler = joblib.load(min_max_scalar_path) 

    target_column_name = "Class"
    X_test, y_test = data.drop(columns=target_column_name,axis="columns"), data[target_column_name].to_numpy()
    

    X_test_scaled = scaler.transform(X_test)
    X_test = xgboost.DMatrix(X_test_scaled.values)
    

    predicted_output_test = model.predict(X_test)
    f1_score_test = metrics.f1_score(y_test,predicted_output_test)

    probas_ = model.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    report_dict = {
        "classification_metrics": {
            "roc_auc": {
                "value": roc_auc,
            },
            "f1_score": {
                "value": f1_score_test,
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))