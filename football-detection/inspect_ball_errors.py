
from pathlib import Path
from collections import defaultdict
import random
import cv2
from ultralytics import YOLO




ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "football"
    / "football_detection_v1-4"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = ROOT / "datasets" / "V1" / "images" / "val"
LABEL_DIR = ROOT / "datasets" / "V1" / "labels" / "val"

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "diagnostics"
    / "ball_errors"
)

BALL_CLASS_ID = 1

CONF_THRESHOLD = 0.05
NMS_IOU_THRESHOLD = 0.70
MATCH_IOU_THRESHOLD = 0.50
MIN_MATCH_IOU = 0.10

MAX_PER_CATEGORY = 30
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}




def read_yolo_labels(label_path, width, height):
    """
    Read YOLO annotations and convert normalized
    coordinates into pixel coordinates.
    """

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as file:

        for line_number, line in enumerate(file, start=1):

            values = line.strip().split()

            if len(values) != 5:
                print(
                    f"Invalid annotation: {label_path.name}, "
                    f"line {line_number}"
                )
                continue

            try:
                cls, xc, yc, bw, bh = map(float, values)
            except ValueError:
                continue

            if not cls.is_integer():
                continue

            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height

            boxes.append({
                "class_id": int(cls),
                "box": [x1, y1, x2, y2]
            })

    return boxes


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between
    two boxes in xyxy format.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)

    intersection = (
        intersection_width * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def draw_box(image, box, color, text):
    """
    Draw a bounding box and its label.
    """

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def save_example(category, filename, image):
    """
    Save an individual diagnostic image.
    """

    output_folder = OUTPUT_DIR / category

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_folder / filename

    cv2.imwrite(
        str(output_path),
        image
    )




def main():

    random.seed(RANDOM_SEED)



    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Validation label directory not found: {LABEL_DIR}"
        )



    print("\nLoading YOLO26n model...")

    model = YOLO(str(MODEL_PATH))

    print(f"Model: {MODEL_PATH}")
    print(f"Ball class ID: {BALL_CLASS_ID}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")


    images = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        raise RuntimeError(
            f"No validation images found in {IMAGE_DIR}"
        )

    print(f"\nFound {len(images)} validation images.")



    categories = defaultdict(list)

    total_gt_balls = 0
    total_predictions = 0
    unreadable_images = 0


    for index, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            unreadable_images += 1
            continue

        height, width = image.shape[:2]

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        # Read annotations.
        labels = read_yolo_labels(
            label_path,
            width,
            height
        )

        gt_balls = [
            item["box"]
            for item in labels
            if item["class_id"] == BALL_CLASS_ID
        ]

        total_gt_balls += len(gt_balls)

        result = model.predict(
            source=image,
            imgsz=640,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=0,
            verbose=False
        )[0]

        pred_balls = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())

                if class_id != BALL_CLASS_ID:
                    continue

                pred_balls.append({
                    "box": box.xyxy[0].cpu().tolist(),
                    "conf": float(box.conf.item()),
                    "matched": False
                })

        total_predictions += len(pred_balls)


        for gt_index, gt_box in enumerate(gt_balls):

            best_index = -1
            best_iou = 0.0

            for pred_index, prediction in enumerate(pred_balls):

                if prediction["matched"]:
                    continue

                score = calculate_iou(
                    gt_box,
                    prediction["box"]
                )

                if score > best_iou:
                    best_iou = score
                    best_index = pred_index

            # Determine category.
            if best_index == -1 or best_iou < MIN_MATCH_IOU:

                category = "missed"

            elif best_iou >= MATCH_IOU_THRESHOLD:

                category = "correct"

                pred_balls[best_index]["matched"] = True

            else:

                category = "poor_localization"

                pred_balls[best_index]["matched"] = True

            example_image = image.copy()

            draw_box(
                example_image,
                gt_box,
                (0, 255, 0),
                "GT Ball"
            )

            if best_index >= 0 and best_iou >= MIN_MATCH_IOU:

                prediction = pred_balls[best_index]

                draw_box(
                    example_image,
                    prediction["box"],
                    (255, 0, 0),
                    (
                        f"Pred {prediction['conf']:.2f} "
                        f"IoU {best_iou:.2f}"
                    )
                )

            filename = (
                f"{image_path.stem}_gt{gt_index}"
                f"{image_path.suffix}"
            )

            categories[category].append(
                (filename, example_image)
            )

        for pred_index, prediction in enumerate(pred_balls):

            if prediction["matched"]:
                continue

            example_image = image.copy()

            draw_box(
                example_image,
                prediction["box"],
                (0, 0, 255),
                f"FP {prediction['conf']:.2f}"
            )

            filename = (
                f"{image_path.stem}_pred{pred_index}"
                f"{image_path.suffix}"
            )

            categories["false_positive"].append(
                (filename, example_image)
            )


        if index % 100 == 0 or index == len(images):

            print(
                f"Processed {index}/{len(images)} images"
            )



    print("\nSaving diagnostic examples...")

    for category in [
        "correct",
        "poor_localization",
        "missed",
        "false_positive"
    ]:

        samples = categories[category]

        random.shuffle(samples)

        output_folder = OUTPUT_DIR / category

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        saved_count = 0

        for filename, result_image in samples[:MAX_PER_CATEGORY]:

            output_path = output_folder / filename

            success = cv2.imwrite(
                str(output_path),
                result_image
            )

            if success:
                saved_count += 1

        print(
            f"{category}: "
            f"{len(samples)} cases | "
            f"{saved_count} images saved"
        )


    print("\n" + "=" * 55)
    print("YOLO26n BALL DETECTION DIAGNOSTICS")
    print("=" * 55)

    print(f"Validation images: {len(images)}")
    print(f"Unreadable images: {unreadable_images}")
    print(f"Ground-truth balls: {total_gt_balls}")
    print(f"Predicted balls: {total_predictions}")

    for category in [
        "correct",
        "poor_localization",
        "missed",
        "false_positive"
    ]:

        print(
            f"{category:20s}: "
            f"{len(categories[category])}"
        )

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print("\nDiagnostic process completed.")


if __name__ == "__main__":
    main()
