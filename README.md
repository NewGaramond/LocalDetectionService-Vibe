# LocalDetectionService-Vibe

Local, containerized **object-detection API**.
**Input:** an image (upload or URL).
**Output:** JSON detections for vehicles & people (mapped to your dataset’s class names).

* Pre-trained YOLO model (no training).
* FastAPI server with `/health`, `/predict`, `/predict/url`.
* Optional validator that runs over your **local dataset** in `data/Vehicles/*`.

---

## 1) Repository & directories

```
LocalDetectionService-Vibe/
├─ app/
│  ├─ main.py          # FastAPI app (routes)
│  ├─ inference.py     # Model init & prediction + class-name mapping
│  └─ schemas.py       # (optional) Pydantic response schema
├─ tests/
│  └─ validate_dataset.py   # Validator that posts your images to the API
├─ scripts/
│  ├─ build.sh / run.sh     # Docker helpers (bash)
│  └─ build.ps1 / run.ps1   # Docker helpers (PowerShell)
├─ .env.example
├─ .gitignore
├─ .dockerignore
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

> **Note:** `data/` is intentionally **git-ignored**. You will create its structure locally.

---

## 2) Data folder scaffold (create locally)

The validator expects images under:

```
data/
└─ Vehicles/
   ├─ Airplanes/
   ├─ Bikes/         # contains MOTORBIKES (see mapping below)
   ├─ Buses/
   ├─ Cars/
   ├─ Helicopters/   # unsupported by the COCO model → skipped 
   ├─ Ships/
   ├─ Trains/
   └─ Trucks/
```

### Quick script to create the empty folder structure

**Option A — Python (portable)**

Save as `scripts/make_data_dirs.py` and run `python scripts/make_data_dirs.py`:

```python
from pathlib import Path

classes = [
    "Airplanes", "Bikes", "Buses", "Cars",
    "Helicopters", "Ships", "Trains", "Trucks",
]

root = Path("data") / "Vehicles"
for c in classes:
    (root / c).mkdir(parents=True, exist_ok=True)
print(f"Created: {root}/*")
```

**Option B — Bash**

```bash
mkdir -p data/Vehicles/{Airplanes,Bikes,Buses,Cars,Helicopters,Ships,Trains,Trucks}
```

**Option C — PowerShell**

```powershell
$classes = "Airplanes","Bikes","Buses","Cars","Helicopters","Ships","Trains","Trucks"
foreach ($c in $classes) {
  New-Item -ItemType Directory -Path "data\Vehicles\$c" -Force | Out-Null
}
```

---

## 3) Dataset used

We validate against the **Vehicles Transportation** dataset from Kaggle:
[https://www.kaggle.com/datasets/aseemks07/vehiclestransport?resource=download-directory](https://www.kaggle.com/datasets/aseemks07/vehiclestransport?resource=download-directory)

**Important alignment notes:**

* The dataset’s **`Bikes`** folder actually contains **motorbikes**.
  The COCO model uses class **`motorcycle`**. Our API **maps** that to **`motorbikes`** in the output for consistency with your dataset wording.
* **Ships** are reported by the model as `boat`; the API maps that to **`ships`**.
* **Airplanes** are reported as `airplane`; the API maps that to **`airplanes`**.
* **Helicopters** are **not in COCO** → the pre-trained model won’t detect them; validator **skips** that class.

Place the downloaded images into the matching folders under `data/Vehicles/*` (keep the capitalization shown above).

---

## 4) What the service does

* Loads a pre-trained YOLO model on startup (CPU by default).
* Accepts an image (upload or URL).
* Runs detection and returns JSON with:

  * `class` (mapped to your dataset names: `car`, `motorbikes`, `airplanes`, `ships`, `buses`, `trucks`, `trains`, `person`)
  * `confidence` (0..1)
  * `bbox` in `xywh` (pixels)
  * `image_size` and `model` metadata

**Sample response**

```json
{
  "model": "yolo11n.pt",
  "image_size": [1280, 720],
  "detections": [
    { "class": "car", "confidence": 0.92, "bbox": { "x": 221, "y": 134, "w": 64, "h": 188 }, "box_format": "xywh" },
    { "class": "motorbikes", "confidence": 0.87, "bbox": { "x": 540, "y": 300, "w": 90, "h": 120 }, "box_format": "xywh" }
  ]
}
```

---

## 5) Run locally (without Docker)

Create a venv, install deps, start the API:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional config (defaults shown)
# export MODEL_WEIGHTS=yolo11n.pt
# export CONF_THRESHOLD=0.25
# export IMGSZ=640
# export ALLOWED_CLASSES="person,car,bicycle,motorcycle,airplane,boat,truck,bus,train"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

* Health: `http://localhost:8000/health`
* Swagger UI: `http://localhost:8000/docs`

**Try a request**

```bash
curl -s -X POST "http://localhost:8000/predict" \
  -F "image=@data/Vehicles/Cars/your_image.jpg" | jq .
```

---

## 6) Run with Docker

### Build

```bash
docker build -t localdetectionservice-vibe:cpu .
```

### Run (PowerShell line breaks use backticks)

```powershell
docker run --rm -p 8000:8000 `
  -e MODEL_WEIGHTS=yolo11n.pt `
  -e CONF_THRESHOLD=0.20 `
  -e IMGSZ=832 `
  -e ALLOWED_CLASSES='person,car,bicycle,motorcycle,airplane,boat,truck,bus,train' `
  localdetectionservice-vibe:cpu
```

**.env option**

```
MODEL_WEIGHTS=yolo11n.pt
CONF_THRESHOLD=0.20
IMGSZ=832
ALLOWED_CLASSES=person,car,bicycle,motorcycle,airplane,boat,truck,bus,train
```

```bash
docker run --rm -p 8000:8000 --env-file .env localdetectionservice-vibe:cpu
```

---

## 7) Validate the service with your local data

With the API running, execute the validator (it posts your local images to the server):

```bash
python tests/validate_dataset.py --root data --per-class 10 --shuffle
# or point to another host:
# python tests/validate_dataset.py --root data --per-class 10 --api http://127.0.0.1:8000
```

You’ll see per-image lines (OK/MISS/SKIP) and a class summary.
**Helicopters** are summarized as **skipped**.

**Tuning tips**

* For small/fast objects (e.g., motorbikes), try:

  * `CONF_THRESHOLD=0.20`
  * `IMGSZ=832` (or `960`)
* Larger `IMGSZ` → better recall (slower).

---

## 8) API reference

* `GET /health` → service status
* `POST /predict` (multipart) → form-data key `image` (file)
* `POST /predict/url` (JSON) → `{ "url": "https://..." }`

---

## 9) Configuration (env)

| Variable          | Default                                                       | Purpose                            |
| ----------------- | ------------------------------------------------------------- | ---------------------------------- |
| `MODEL_WEIGHTS`   | `yolo11n.pt`                                                  | YOLO weights to load               |
| `CONF_THRESHOLD`  | `0.25`                                                        | Min confidence to keep a detection |
| `IMGSZ`           | `640`                                                         | Inference image size (long side)   |
| `ALLOWED_CLASSES` | `person,car,bicycle,motorcycle,airplane,boat,truck,bus,train` | Class filter (COCO names)          |

> Output classes are mapped to your dataset terms:
> `motorcycle → motorbikes`, `boat → ships`, `airplane → airplanes`.
> `helicopters` are unsupported by the base COCO model.

---

## 10) Notes & license

* This repo uses the **Ultralytics YOLO** Python package and pre-trained weights.
  Review licensing if you plan to redistribute or use commercially.
* Dataset images are **not** included; download from Kaggle and place under `data/Vehicles/*`.

---

## 11) Troubleshooting

* **404 on `/`**: use `/health` or `/docs` (we don’t define a root route).
* **PowerShell command wrapping**: use backticks `` ` `` for line continuation (no trailing spaces).
* **Weights download in Docker build fails**: comment out the pre-download step; weights will fetch on first inference.

---

Happy detecting!
