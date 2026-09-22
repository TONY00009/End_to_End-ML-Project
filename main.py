from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/predictdata")
async def predict_datapoint_get(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.post("/predictdata")
async def predict_datapoint_post(
    request: Request,
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    
    pred_df = data.get_data_as_data_frame()
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    # Pass variables to the HTML using the context dictionary
    return templates.TemplateResponse(request=request, name="home.html", context={"results": results[0]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)