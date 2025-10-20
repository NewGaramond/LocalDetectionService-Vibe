import os, argparse, json, random
from pathlib import Path
from collections import defaultdict
import httpx

# Folder → expected label (matches API output mapping)
FOLDER_TO_LABEL = {
    "Airplanes": "airplanes",
    "Bikes": "motorbikes",       # <-- changed (was bike)
    "MotorBikes": "motorbikes",  # <-- handle this variant too
    "Buses": "buses",
    "Cars": "car",
    "Helicopters": "helicopters",  # still skipped as unsupported
    "Ships": "ships",
    "Trains": "trains",
    "Trucks": "trucks",
}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def iter_images(vehicles_dir: Path):
    if not vehicles_dir.exists():
        raise SystemExit(f"Not found: {vehicles_dir} (expected 'data/Vehicles')")
    for class_dir in sorted(p for p in vehicles_dir.iterdir() if p.is_dir()):
        cls_name = class_dir.name  # keep original capitalization
        for img in class_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                yield cls_name, img

def call_api(api_base: str, image_path: Path):
    url = api_base.rstrip("/") + "/predict"
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "application/octet-stream")}
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            r = client.post(url, files=files)
            r.raise_for_status()
            return r.json()

def main():
    ap = argparse.ArgumentParser(description="Validate detections over data/Vehicles/*")
    ap.add_argument("--root", default="data", help="Root folder containing Vehicles/")
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--per-class", type=int, default=10, help="Max images per class (0 = no limit)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle images before limiting per class")
    ap.add_argument("--save-json", default="", help="Optional path to save raw results JSON")
    args = ap.parse_args()

    vehicles_dir = Path(args.root) / "Vehicles"
    totals = defaultdict(int)
    hits = defaultdict(int)
    skipped = defaultdict(int)
    results_dump = []

    # Collect images per class first (so we can limit/shuffle)
    bucket = defaultdict(list)
    for cls_folder, img in iter_images(vehicles_dir):
        bucket[cls_folder].append(img)

    # Process class by class
    for cls_folder, imgs in bucket.items():
        if cls_folder not in FOLDER_TO_LABEL:
            print(f" - WARN: Unknown folder '{cls_folder}', skipping its images")
            continue

        expected = FOLDER_TO_LABEL[cls_folder]
        imgs_use = imgs[:]
        if args.shuffle:
            random.shuffle(imgs_use)
        if args.per_class > 0:
            imgs_use = imgs_use[:args.per_class]

        if expected == "helicopters":
            # unsupported by the YOLO COCO model we use → skip
            skipped[cls_folder] += len(imgs_use)
            for img_path in imgs_use:
                print(f"SKIP [{cls_folder}] {img_path.name}  (unsupported 'helicopters')")
            continue

        for img_path in imgs_use:
            try:
                payload = call_api(args.api, img_path)
            except Exception as e:
                print(f"ERR  [{cls_folder}] {img_path.name} → API error: {e}")
                continue

            dets = payload.get("detections", [])
            found = any(d.get("class") == expected for d in dets)
            preview = sorted(dets, key=lambda d: d.get("confidence", 0), reverse=True)[:3]
            pv = ", ".join(f"{d['class']}:{d['confidence']:.2f}" for d in preview) if preview else "—"

            totals[cls_folder] += 1
            if found:
                hits[cls_folder] += 1
                print(f"OK   [{cls_folder}] {img_path.name} → found '{expected}'  | {pv}")
            else:
                print(f"MISS [{cls_folder}] {img_path.name} → no '{expected}'  | {pv}")

            results_dump.append({
                "class_folder": cls_folder,
                "image": img_path.as_posix(),
                "expected": expected,
                "found": found,
                "top3": preview,
                "raw_count": len(dets)
            })

    # Summary
    print("\n=== SUMMARY ===")
    all_tot = sum(totals.values())
    all_hit = sum(hits.values())
    for cls in sorted(totals.keys()):
        t = totals[cls]
        h = hits[cls]
        rate = (h / t * 100) if t else 0.0
        print(f"{cls:12s}: {h:3d}/{t:3d}  ({rate:5.1f}%)  skipped:{skipped.get(cls,0)}")
    print(f"\nTOTAL: {all_hit}/{all_tot}  ({(all_hit/all_tot*100 if all_tot else 0):.1f}%)  | skipped total: {sum(skipped.values())}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "summary": {
                "totals": totals,
                "hits": hits,
                "skipped": skipped,
                "all_total": all_tot,
                "all_hits": all_hit
            },
            "results": results_dump
        }, indent=2, default=lambda o: dict(o)), encoding="utf-8")
        print(f"\nSaved raw results to: {out_path}")

if __name__ == "__main__":
    main()
