from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"message": "I like to move it"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
#внесены изменения для пулл реквеста