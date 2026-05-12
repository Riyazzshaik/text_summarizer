from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
import os

from textSummarizer.pipeline.prediction import PredictionPipeline


# CREATE FASTAPI APP
app = FastAPI(
    title="Text Summarizer API",
    version="1.0"
)


# INPUT MODEL
class TextRequest(BaseModel):
    text: str


# HOME ROUTE
@app.get("/", tags=["Home"])
async def home():
    return RedirectResponse(url="/docs")


# TRAINING ROUTE
@app.get("/train", tags=["Training"])
async def train_model():

    try:
        os.system("python main.py")

        return {
            "message": "Training completed successfully"
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# PREDICTION ROUTE
@app.post("/predict", tags=["Prediction"])
async def predict_route(request: TextRequest):

    try:

        obj = PredictionPipeline()

        summary = obj.predict(request.text)

        return {
            "summary": summary
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# MAIN
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )