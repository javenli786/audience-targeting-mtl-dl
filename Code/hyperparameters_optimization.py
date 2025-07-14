import optuna
import mlflow
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from Model_Building import create_model, train_model
from MLflow_Setup import log_results

def objective(trial,data_dict):
    """
    Set up the objective function for Optuna hyperparameter optimization.
    """
    X_train = data_dict['X_train']
    Y_regression_train = data_dict['Y_regression_train']
    Y_classification_train = data_dict['Y_classification_train']
    X_val = data_dict['X_val']
    Y_regression_val = data_dict['Y_regression_val']
    Y_classification_val = data_dict['Y_classification_val']
    
    with mlflow.start_run(nested=True) as run:
        params = {
            "units": trial.suggest_categorical("units", [16, 32, 64, 128]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "elu", "tanh"])
        }
        
        trial_num = trial.number
        mlflow.set_tag("trial_num", trial_num)
        mlflow.log_params({**params, "batch_size": 32, "epochs": 50, "run_id": run.info.run_id})
        
        model = create_model(params, X_train.shape[1:])
        
        class_weights = compute_class_weight('balanced', classes=Y_classification_train.unique(), y=Y_classification_train)
        weight_map = {i: w for i, w in zip(Y_classification_train.unique(), class_weights)}
        sample_weights = np.array([weight_map[y] for y in Y_classification_train])
        
        # Train the model
        train_targets = {'regression_output': Y_regression_train, 'classification_output': Y_classification_train}
        val_targets = {'regression_output': Y_regression_val, 'classification_output': Y_classification_val}
        history = train_model(model, X_train, train_targets, X_val, val_targets, sample_weights)
        
        # log results
        log_results(trial, run, history, model, trial_num)
        
        # Generate the best recall metric
        best_recall = max(history.history['val_classification_output_recall'])
        return best_recall
    
    

def optimize_hyperparameters(n_trials,data_dict):
    """
    Optimize hyperparameters Optuna.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M")
    
    with mlflow.start_run(run_name=f"Optuna_Optimization_{now}") as main_run:
        # Create the Optuna study
        study = optuna.create_study(direction="maximize", study_name="Optuna_Optimization")
        study.optimize(lambda trial:objective(trial,data_dict), n_trials=n_trials)
        
        # Log the best trial information
        best_trial = study.best_trial
        best_params = study.best_params
        
        mlflow.log_params({
            "best_trial": best_trial.number,
            "best_recall": study.best_value,
            "total_trials": len(study.trials)
        })
        
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        # Log the best trial metrics
        best_metrics = {k: v for k, v in best_trial.user_attrs.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(best_metrics)
        
        # Log the best model
        best_model_uri = f"runs:/{best_trial.user_attrs['run_id']}/{best_trial.user_attrs['model_path']}"
        best_model = mlflow.keras.load_model(best_model_uri)
        
        mlflow.keras.log_model(
            keras_model=best_model, 
            artifact_path="best_model",
            registered_model_name="DonationMaximization_MTL_Model"
        )
        
        return best_model, best_params, study.best_value
