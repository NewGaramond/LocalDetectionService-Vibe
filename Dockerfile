FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# deps for OpenCV backends used by ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) pre-download weights so first run is offline-fast.
# If your build env blocks outbound internet, comment this block.
RUN python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11n.pt")
PY

# App code
COPY app app

# (tests are optional; copy if you want to run validator against the container)
COPY tests tests

EXPOSE 8000

# Defaults — you can override at `docker run -e ...`
ENV MODEL_WEIGHTS=yolo11n.pt \
    CONF_THRESHOLD=0.25 \
    IMGSZ=640 \
    ALLOWED_CLASSES=person,car,bicycle,motorcycle,airplane,boat,truck,bus,train

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
