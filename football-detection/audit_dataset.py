
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "V1"

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "Player",
    1: "Ball",
    2: "Referee"
}


def main():
    for split in SPLITS:
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        if not image_dir.exists() or not label_dir.exists():
            print(f"\nSkipping missing split: {split}")
            continue

        image_paths = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        counts = Counter()
        missing_labels = []
        invalid_labels = []
        ball_sizes = []
        image_sizes = []

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_sizes.append((width, height))
            except Exception as exc:
                print(f"Unreadable image: {image_path} ({exc})")
                continue

            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue

            with open(label_path, "r") as file:
                for line_number, line in enumerate(file, start=1):
                    values = line.strip().split()

                    if len(values) != 5:
                        invalid_labels.append(
                            (label_path.name, line_number, "Expected 5 values")
                        )
                        continue

                    try:
                        cls, xc, yc, bw, bh = map(float, values)
                    except ValueError:
                        invalid_labels.append(
                            (label_path.name, line_number, "Non-numeric value")
                        )
                        continue

                    if not cls.is_integer() or int(cls) not in CLASS_NAMES:
                        invalid_labels.append(
                            (label_path.name, line_number, "Invalid class ID")
                        )
                        continue

                    if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                        invalid_labels.append(
                            (label_path.name, line_number, "Coordinate out of range")
                        )
                        continue

                    class_id = int(cls)
                    counts[class_id] += 1

                    if class_id == 1:
                        ball_sizes.append((bw, bh))

        print(f"\n========== {split.upper()} ==========")
        print(f"Images: {len(image_paths)}")
        print(f"Missing label files: {len(missing_labels)}")
        print(f"Invalid annotations: {len(invalid_labels)}")

        for class_id, name in CLASS_NAMES.items():
            print(f"{name}: {counts[class_id]}")

        if ball_sizes:
            areas = [w * h for w, h in ball_sizes]
            print(f"Ball annotations: {len(ball_sizes)}")
            print(f"Average normalized ball box area: {sum(areas)/len(areas):.6f}")
            print(f"Smallest normalized ball box area: {min(areas):.6f}")
            print(f"Largest normalized ball box area: {max(areas):.6f}")

        if image_sizes:
            print(f"Image dimensions (first 5): {image_sizes[:5]}")

        if missing_labels:
            print("Missing label examples:", missing_labels[:5])

        if invalid_labels:
            print("Invalid label examples:", invalid_labels[:5])


if __name__ == "__main__":
    main()
