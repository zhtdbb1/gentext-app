from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from fastapi import (
    FastAPI,
    HTTPException,
    File,
    UploadFile,
)
from app.image_classifier import ImageClassifier


app = FastAPI()
embedding_model = EmbeddingModel()
image_classifier = ImageClassifier()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]


# Create the bigram model
bigram_model = BigramModel(corpus)


# Request format for the /generate endpoint
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {"generated_text": generated_text}


class EmbeddingRequest(BaseModel):
    word: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    try:
        return embedding_model.get_embedding(request.word)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error



@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    max_size = 10 * 1024 * 1024
    image_bytes = file.file.read(max_size + 1)

    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail="Upload an image smaller than 10 MB.",
        )

    try:
        return image_classifier.predict(image_bytes)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error