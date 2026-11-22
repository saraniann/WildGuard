
from pathlib import Path
import argparse
import yaml
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SOURCE_YAML = DATASET / "data.yaml"
LOCAL_YAML = DATASET / "data_local.yaml"

EXPECTED_CLASSES = [
    "cow", "monkey", "tapir", "dog", "person"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=2
    )
    args = parser.parse_args()

    # Verify exported dataset.
    if not SOURCE_YAML.exists():
        raise FileNotFoundError(
            f"Missing dataset YAML: {SOURCE_YAML}"
        )

    for split in ["train", "valid", "test"]:
        for folder in ["images", "labels"]:
            path = DATASET / split / folder
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Missing dataset folder: {path}"
                )

    # Read dataset configuration.
    with open(SOURCE_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    actual_classes = [
        names[i] if isinstance(names, list)
        else names.get(i, names.get(str(i)))
        for i in range(5)
    ]

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Incorrect class mapping: {actual_classes}"
        )

    # Use an absolute path for reliable training.
    data["path"] = DATASET.resolve().as_posix()

    with open(LOCAL_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data, f, sort_keys=False
        )

    print("Dataset:", DATASET)
    print("Classes:", actual_classes)
    print("CUDA GPU available:", torch.cuda.is_available())

    # Load pretrained YOLO11 nano weights.
    model = YOLO("yolo11n.pt")

    # Separate quick tests from full training.
    run_name = (
        "smoke_test"
        if args.epochs <= 2
        else "first_training"
    )

    model.train(
    data=str(LOCAL_YAML),
    epochs=args.epochs,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    patience=50,
    amp=True,
    project=str(ROOT / "runs" / "wildguard"),
    name="gpu_200_epochs",
    exist_ok=False,
)

    print("Training finished.")
    print("Run:", run_name)

if __name__ == "__main__":
    main()
