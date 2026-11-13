from ultralytics import YOLO
from pathlib import Path
import torch


def main():

    ROOT = Path(__file__).resolve().parent
    DATA_YAML = ROOT / "datasets" / "V1" / "data.yaml"


    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing Dataset: {DATA_YAML}")



    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is unavailable, Check the pyTorch installation")


    
    print("="*50)
    print("Football Detection Training")
    print("="*50)

    
    print("GPU: ", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")
    print("Dataset: ", DATA_YAML)


    model = YOLO("yolo26n.pt")

    result = model.train(
        data = str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        patience=20,
        amp = True, 
        optimizer = "auto", 
        project = str(ROOT / "runs" / "football"),
        name = "football_detection_v1", 
        save = True, 
        save_period = 10, 
        plots = True,
        seed = 42, 
        deterministic = True, 
        verbose = True

    )


    print("\nTraining completed!")
    print("Best model:", ROOT / "runs" / "football" /
          "yolo26n_baseline" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
