import mlflow
import argparse
import pandas as pd
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from Data_Preparation import prepare_data
from MLflow_Setup import setup_mlflow, log_results
from Model_Building import create_model, train_model
from Hyperparameters_Optimization import optimize_hyperparameters
from Model_Prediction import generate_predictions
from Model_Registry import register_model

TRACKING_URI = 'http://192.168.1.162:5000'
EXPERIMENT_NAME = 'DonationMaximization'
MODEL_NAME = 'DonationMaximization_MTL_Model'

def main_pipeline(retrain=True, n_trials=5):
    if retrain:

        setup_mlflow(TRACKING_URI, EXPERIMENT_NAME)
        data_dict = prepare_data(retrain=True)
        best_model,best_params, best_value = optimize_hyperparameters(n_trials,data_dict)
        register_model(best_model)
        return best_model, best_params, best_value
        
    else:
        setup_mlflow(TRACKING_URI, EXPERIMENT_NAME)
        data_dict=prepare_data(retrain=False)
        predictions=generate_predictions(MODEL_NAME, data_dict['X_test'])
        return predictions
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DonationMaximization")
    parser.add_argument("--retrain", action="store_true", help="Retrain model")
    parser.add_argument("--trials", type=int, default=5, help="Optimization trials")
    
    args = parser.parse_args()
    
    result = main_pipeline(retrain=args.retrain, n_trials=args.trials)
        
