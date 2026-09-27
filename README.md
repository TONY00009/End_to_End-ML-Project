# End-to-End Student Math Score Predictor | MLOps Pipeline

An end-to-end machine learning system that predicts a student's math score based on demographic and academic indicators (gender, ethnicity, parental education, lunch type, test preparation, and reading/writing scores). 

Beyond standard local model training, this project implements a cloud-native **MLOps architecture** featuring decoupled remote experiment tracking on AWS (EC2, RDS, S3), Docker containerization, and automated CI/CD deployment via GitHub Actions.

---

## System Architecture

```text
  [ Local Development (VS Code) ]
                 |
                 |---> 1. Runs Training Pipeline (src/components/data_ingestion.py)
                 |        |-- Evaluates 7 Regression Models
                 |        |-- Logs Metrics & Hyperparameters ---> [ AWS RDS (PostgreSQL) ]
                 |        |-- Streams Model Artifacts ----------> [ AWS S3 Bucket ]
                 |        |-- Saves Best Winner Locally --------> (artifacts/model.pkl)
                 |
                 |---> 2. Git Push Triggers CI/CD (GitHub Actions)
                          |-- Builds Docker Image & Pushes -----> [ Amazon ECR ]
                          |-- SSH into Cloud Server ------------> [ AWS EC2 Instance ]
                                                                     |-- Port 5000: MLflow Tracking UI
                                                                     |-- Port 8000: FastAPI Prediction App
