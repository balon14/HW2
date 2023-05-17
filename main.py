from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


# Заголовок страницы
st.title("A Simple Streamlit Web App")

# Получение имени пользователя и вывод приветствия
text = st.text_input("Enter your name", "")
st.write(f"Hello {text}!")


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"message": "Hello world"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
