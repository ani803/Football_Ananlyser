
from pathlib import Path
from collections import Counter
import cv2
import csv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "runs" / "football" / "football_detection_v1-4"
    / "weights" / "best.pt"
)

IMAGE_DIR = ROOT / "datasets" / "V1" / "images" / "val"
LABEL_DIR = ROOT / "datasets" / "V1" / "labels" / "val"

OUTPUT_DIR = ROOT / "runs" / "diagnostics" / "prediction_audit"

BALL_CLASS_ID = 1
CONF_THRESHOLD = 0.05

def has_gt_ball(label_path):
    if not label_path.exists():
        return False

    for line in label_path.read_text().splitlines():
        values = line.split()
        if values and values[0] == str(BALL_CLASS_ID):
            return True

    return False


def main():
    model = YOLO(str(MODEL_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "ball_predictions.csv"

    summary = Counter()
    rows = []

    images = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    for index, image_path in enumerate(images, start=1):

        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        annotated = has_gt_ball(label_path)

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        result = model.predict(
            source=image,
            imgsz=640,
            conf=CONF_THRESHOLD,
            iou=0.70,
            device=0,
            verbose=False
        )[0]

        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) != BALL_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                conf = float(box.conf.item())

                predictions.append((x1, y1, x2, y2, conf))

        group = (
            "annotated_ball"
            if annotated
            else "no_annotated_ball"
        )

        summary[group] += 1
        summary[f"{group}_predictions"] += len(predictions)

        for pred_index, (x1, y1, x2, y2, conf) in enumerate(predictions):

            width = x2 - x1
            height = y2 - y1

            rows.append({
                "image": image_path.name,
                "group": group,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_width": width,
                "box_height": height,
                "box_area": width * height
            })

            # Save predictions from images without annotated balls.
            if not annotated:
                output = image.copy()

                cv2.rectangle(
                    output,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    output,
                    f"Ball {conf:.2f}",
                    (int(x1), max(20, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                folder = OUTPUT_DIR / "no_annotated_ball"
                folder.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(
                    str(folder / f"{image_path.stem}_{pred_index}.jpg"),
                    output
                )

        if index % 100 == 0:
            print(f"Processed {index}/{len(images)}")

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print("\n========== PREDICTION AUDIT ==========")
    print(f"Images with annotated balls: {summary['annotated_ball']}")
    print(f"Images without annotated balls: {summary['no_annotated_ball']}")
    print(f"Predictions in annotated-ball images: {summary['annotated_ball_predictions']}")
    print(f"Predictions in images without annotated balls: {summary['no_annotated_ball_predictions']}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Visual examples saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
