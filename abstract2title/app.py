import torch
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from simpletransformers.seq2seq import Seq2SeqModel
from starlette.requests import Request

from abstract2title.paths import ROOT_DIR

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained BART model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name=ROOT_DIR / "data" / "model_checkpoints" / "2025-11-12",
    use_cuda=True if torch.cuda.is_available() else False,
)

# Initialize Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory="templates")


# Define the home page route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Define the prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, abstract: str = Form(...)):
    # Error handling to ensure the input is a valid string
    if not isinstance(abstract, str) or not abstract.strip():
        error_message = "Invalid input. Please enter a valid abstract as text."
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": error_message}
        )

    try:
        # Make a prediction using the model
        predicted_title = model.predict([abstract.strip()])[0]
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": error_message}
        )

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "abstract": abstract, "predicted_title": predicted_title},
    )


# Run the FastAPI app with `uvicorn` using: `uvicorn app:app --reload`
