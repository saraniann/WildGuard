
from pathlib import Path
from ultralytics import YOLO

# Load the pretrained YOLO11 nano model.
# The weights download automatically on the first run.
model = YOLO("yolo11n.pt")

# COCO class IDs for our Phase 1 baseline:
# 0 = person, 16 = dog, 19 = cow
TARGET_CLASSES = [0, 16, 19]

# Official sample image containing people and a bus.
source = "https://ultralytics.com/images/bus.jpg"

results = model.predict(
    source=source,
    classes=TARGET_CLASSES,
    conf=0.25,
    save=True,
    project="runs",
    name="first_detection",
    exist_ok=True,
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        print(
            f"Detected: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

print("Detection completed.")
print("Check the runs/first_detection folder.")
