# You can tweak values here or pass them via --env-file .env
docker run --rm -p 8000:8000 `
  -e MODEL_WEIGHTS=yolo11n.pt `
  -e CONF_THRESHOLD=0.20 `
  -e IMGSZ=832 `
  -e ALLOWED_CLASSES='person,car,bicycle,motorcycle,airplane,boat,truck,bus,train' `
  localdetectionservice-vibe:cpu
