from fastapi import FastAPI
from pydantic import BaseModel

# Import the prediction logic from our model.py
from app.model import predict_phishing

# Create the FastAPI application
app = FastAPI()

# Define the data model for the request body using Pydantic
# This ensures the input data is a JSON with a "url" key
class URLItem(BaseModel):
    url: str

# Define the root endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Phishing Detector API is running."}


# Define the prediction endpoint
@app.post("/predict")
def make_prediction(item: URLItem):
    """
    Accepts a URL and returns a phishing prediction.
    """
    prediction = predict_phishing(item.url)
    
    return {
        "url": item.url,
        "prediction": prediction
    }