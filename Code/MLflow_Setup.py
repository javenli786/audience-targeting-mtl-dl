import mlflow

def setup_mlflow(TRACKING_URI='http://192.168.1.162:5000', EXPERIMENT_NAME='DonationMaximization'):
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME) 
    else:
        experiment_id = experiment.experiment_id



def log_results(trial, run, history, model, trial_num):
    """
    Log the results of a trial to MLflow.
    Args:
        trial: The Optuna trial object.
        run: The MLflow run object.
        history: The training history of the model.
        model: The model.
        trial_num: The trial number for logging purposes.
    """
    # Log metrics
    for epoch in range(len(history.history['loss'])):
        metrics_dict = {
            metric_name: values[epoch] 
            for metric_name, values in history.history.items() 
            if epoch < len(values)
        }
        mlflow.log_metrics(metrics_dict, step=epoch)
    
    # Set user attributes for the trial
    trial.set_user_attr("run_id", run.info.run_id)
    model_path = f"model_trial_{trial_num}"
    trial.set_user_attr("model_path", model_path)
    
    for metric_name, value in metrics_dict.items():
        trial.set_user_attr(metric_name, value)
    
    # Log the model
    mlflow.keras.log_model(keras_model=model, artifact_path=model_path)
    
    return metrics_dict