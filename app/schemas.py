from pydantic import BaseModel, Field
from typing import List, Literal

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class Detection(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float
    bbox: BBox
    box_format: Literal["xywh"] = "xywh"

class PredictResponse(BaseModel):
    model: str
    image_size: List[int]  # [width, height]
    detections: List[Detection]
