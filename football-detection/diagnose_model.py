from pathlib import Path
from ultralytics import YOLO
import torch 


ROOT = Path(__file__).resolve().parent


DATA_YAML = ROOT / "datasets" / "V1" / "data.yaml"

MODEL_PATH = (
    ROOT
    / "runs"
    / "football"
    / "football_detection_v1-4"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "runs" / "diagnostics"

def main(): 
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    if not MODEL_PATH.exists(): 
        raise FileNotFoundError(f"Model Checkpoint not found: {MODEL_PATH}")

    print("loading Model YOLO 26n......")
    model = YOLO(str(MODEL_PATH))

    print(f"Cuda Available: {torch.cuda.is_available()}")


    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print("\nRunning validation diagnostics...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        save_json=False,
        project=str(OUTPUT_DIR),
        name="validation_diagnostics",
        exist_ok=True
    )

    print("\n========== OVERALL RESULTS ==========")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    print("\n========== CLASS-WISE RESULTS ==========")

    for class_id, class_name in metrics.names.items():
        precision = metrics.box.p[class_id]
        recall = metrics.box.r[class_id]
        map50 = metrics.box.ap50[class_id]
        map5095 = metrics.box.ap[class_id].mean()

        print(f"\n{class_name}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  mAP50:     {map50:.4f}")
        print(f"  mAP50-95:  {map5095:.4f}")

    print("\nDiagnostic output:")
    print(metrics.save_dir)


if __name__ == "__main__":
    main()