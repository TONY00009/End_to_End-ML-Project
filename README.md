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
```

---

## Tech Stack

* **Machine Learning & Data Processing:** Python, Scikit-Learn, CatBoost, XGBoost, Pandas, NumPy
* **Experiment Tracking:** MLflow
* **Web Framework & Serving:** FastAPI, Uvicorn, HTML/CSS (Jinja2)
* **Cloud Infrastructure (AWS):** 
  * **EC2 (Ubuntu):** Hosts the live FastAPI web server and the MLflow tracking daemon
  * **RDS (PostgreSQL):** Stores persistent MLflow experiment metadata and metrics
  * **S3:** Remote artifact store for serialized models
  * **ECR:** Private container registry for Docker images
* **CI/CD & DevOps:** Docker, GitHub Actions

---

## Model Evaluation & Selection

During training, `model_trainer.py` evaluates seven regression algorithms and automatically selects and serializes the top-performing model (`artifacts/model.pkl`) based on the test R² score:

| Algorithm | R² Score | Status |
| :--- | :---: | :--- |
| **Linear Regression** | **0.880** | **Deployed (Best Model)** |
| Gradient Boosting | 0.875 | Logged in MLflow |
| CatBoosting Regressor | 0.861 | Logged in MLflow |
| AdaBoost Regressor | 0.854 | Logged in MLflow |
| Random Forest | 0.853 | Logged in MLflow |
| XGBRegressor | 0.849 | Logged in MLflow |
| Decision Tree | 0.767 | Logged in MLflow |

---

## Project Structure

```text
├── .github/workflows/
│   └── main.yaml              # CI/CD pipeline for ECR build & EC2 deployment
├── artifacts/                 # Serialized preprocessor.pkl, model.pkl, and train/test splits
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Reads raw data and creates train/test splits
│   │   ├── data_transformation.py # Feature scaling & one-hot encoding pipelines
│   │   └── model_trainer.py       # Trains models, logs to AWS MLflow, saves best model
│   ├── pipeline/
│   │   └── predict_pipeline.py    # Loads artifacts/model.pkl for live web inference
│   ├── exception.py           # Custom exception handling
│   ├── logger.py              # Execution logging
│   └── utils.py               # Helper functions (save_object, load_object, evaluate_models)
├── templates/                 # HTML frontend forms for FastAPI
├── Dockerfile                 # Container specification
├── main.py                    # FastAPI application entry point
└── requirements.txt           # Project dependencies
```

---

## Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TONY00009/End_to_End-ML-Project.git](https://github.com/TONY00009/End_to_End-ML-Project.git)
   cd End_to_End-ML-Project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv project_venv
   project_venv\Scripts\activate   # Windows
   # source project_venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training pipeline:**
   ```bash
   python src/components/data_ingestion.py
   ```

5. **Start the FastAPI web application:**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   Open `http://127.0.0.1:8000/predictdata` in your browser to generate predictions.
