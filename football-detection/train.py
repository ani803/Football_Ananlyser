from ultralytics import YOLO
from pathlib import Path
import torch


def main():

    project_dir = Path(__file__).resolve().parent
    data_yaml = project_dir / "datasets" / "V1" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing Dataset: {data_yaml}")

    
    print("GPU: ", torch.cuda.get_device_name(0))
    print("Dataset: ", data_yaml)


    model = YOLO("yolo26n.pt")

    model.train(
        data = "datasets/V1/data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 8,
        device = 0, 
        workers = 8, 
        patience = 20, 
        amp = True, 
        project = "runs/football",
        name = "football_detection_v1", 
        save = True, 
        plots = True

    )


if __name__ == "__main__":
    main()