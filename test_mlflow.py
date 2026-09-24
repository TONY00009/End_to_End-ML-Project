import mlflow

# Connect to your active AWS server
mlflow.set_tracking_uri("http://52.87.173.61:5000")

# Create a clean experiment name
mlflow.set_experiment("AWS_Deployment_Test")

with mlflow.start_run():
    # Log configuration parameters
    mlflow.log_param("algorithm", "Gradient Descent")
    mlflow.log_param("learning_rate", 0.01)
    
    # Log metrics over "time" (or steps) to build a chart
    for step, acc in enumerate([0.75, 0.82, 0.88, 0.91, 0.94]):
        mlflow.log_metric("accuracy", acc, step=step)
        
    print("Data successfully sent to AWS!")