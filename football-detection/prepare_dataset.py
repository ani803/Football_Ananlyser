from pathlib import Path
import random 
import shutil


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "V1"

IMAGE_TRAIN = DATASET / "images" / "train"
LABEL_TRAIN = DATASET / "labels" / "train"

IMAGE_VAL = DATASET / "images" / "val"
LABEL_VAL = DATASET / "labels" / "val"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if IMAGE_VAL.exists() or LABEL_VAL.exists():
        raise FileExistsError("Validation Folders alrady exists, check them before procedding ")
    images = [
        p for p in IMAGE_TRAIN.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise RuntimeError("No Training Images found.")


    missing = [
        image.name for image in images
        if not (LABEL_TRAIN / f"{image.stem}.txt").exists()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} images have missing labels"
            f"Examples: {missing[:5]}"
        )


    random.seed(42)
    random.shuffle(images)


    val_count = round(len(images) * 0.20)
    val_images = images[:val_count]

    #validation directories
    IMAGE_VAL.mkdir(parents=True)
    LABEL_VAL.mkdir(parents=True)

    for image in val_images:
        label = LABEL_TRAIN / f"{image.stem}.txt"

        shutil.move(str(image), str(IMAGE_VAL / image.name))
        shutil.move(str(label), str(LABEL_VAL / label.name))
    print("\nDataset split completed!")
    print("Training images:", len(images) - val_count)
    print("Validation images:", val_count)
    print("Validation labels:", len(list(LABEL_VAL.glob("*.txt"))))

if __name__ == "__main__":
    main()
