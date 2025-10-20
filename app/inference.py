import io
import os
from typing import Dict, Any, List
from PIL import Image
import httpx

_MODEL = None
_MODEL_NAME = None

# COCO → your dataset label mapping (output side)
_ALIAS_OUT = {        # if you ever add true bicycles later
    "motorcycle": "motorbikes", # <-- add this line
    "boat": "ships",
    "airplane": "airplanes",
    "bus": "buses",
    "truck": "trucks",
    "train": "trains",
    "car": "car",
    "person": "person",
}

def load_model():
    """Load Ultralytics YOLO once at startup."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL
    weights = os.getenv("MODEL_WEIGHTS", "yolo11n.pt")
    from ultralytics import YOLO
    _MODEL = YOLO(weights)
    _MODEL_NAME = getattr(_MODEL, "ckpt_path", None) or weights
    return _MODEL

def _allowed_classes() -> set:
    # Use COCO names for filtering (pre-alias)
    raw = os.getenv("ALLOWED_CLASSES",
                    "person,car,bicycle,motorcycle,airplane,boat,truck,bus,train")
    return set(x.strip().lower() for x in raw.split(",") if x.strip())

def _conf_threshold() -> float:
    try:
        return float(os.getenv("CONF_THRESHOLD", "0.25"))
    except ValueError:
        return 0.25

def _imgsz() -> int:
    try:
        return int(os.getenv("IMGSZ", "640"))  # try 832 for better small-object recall
    except ValueError:
        return 640

def _label_out(cls_name: str) -> str:
    return _ALIAS_OUT.get(cls_name, cls_name)

def _to_xywh(xyxy, width: int, height: int):
    x1, y1, x2, y2 = xyxy
    x = max(0, int(round(x1)))
    y = max(0, int(round(y1)))
    w = max(0, int(round(x2 - x1)))
    h = max(0, int(round(y2 - y1)))
    w = min(w, max(0, width - x))
    h = min(h, max(0, height - y))
    return {"x": x, "y": y, "w": w, "h": h}

def _run_inference(pil_img: Image.Image) -> Dict[str, Any]:
    model = load_model()
    conf = _conf_threshold()
    allowed = _allowed_classes()

    results = model.predict(pil_img, conf=conf, imgsz=_imgsz(), verbose=False)
    result = results[0]
    names = result.names
    width, height = pil_img.size

    detections: List[Dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_idx = int(box.cls[0].item())
            cls_name = names.get(cls_idx, str(cls_idx)).lower()
            if cls_name not in allowed:
                continue
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": _label_out(cls_name),     # mapped to your dataset terms
                "confidence": round(score, 4),
                "bbox": _to_xywh((x1, y1, x2, y2), width, height),
                "box_format": "xywh"
            })

    return {"model": _MODEL_NAME, "image_size": [width, height], "detections": detections}

def predict_bytes(content: bytes) -> Dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}")
    return _run_inference(img)

def predict_from_url(url: str) -> Dict[str, Any]:
    timeout = httpx.Timeout(10.0, connect=5.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return predict_bytes(resp.content)
