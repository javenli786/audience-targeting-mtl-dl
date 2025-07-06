def Setup_MLflow():
    import mlflow

    TRACKING_URI = "http://127.0.0.1:5000"
    EXPERIMENT_NAME = "DonationMaximization"

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME) 
    else:
        experiment_id = experiment.experiment_id