from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from app.inference import load_model, predict_bytes, predict_from_url

app = FastAPI(title="LocalDetectionService-Vibe", version="0.1.0")

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "service": "LocalDetectionService-Vibe"}

class UrlPayload(BaseModel):
    url: HttpUrl

# Multipart upload
@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="No image provided.")
    content = await image.read()
    result = predict_bytes(content)
    return JSONResponse(result)

# URL input
@app.post("/predict/url")
async def predict_image_url(payload: UrlPayload):
    result = predict_from_url(str(payload.url))
    return JSONResponse(result)
