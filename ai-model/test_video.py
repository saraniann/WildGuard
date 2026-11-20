
from pathlib import Path
from ultralytics import YOLO

# Determine the project root.
ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "testing" / "test.mp4"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

# Load pretrained detector.
model = YOLO("yolo11n.pt")

# Person, Dog, Cow
TARGET_CLASSES = [0, 16, 19]

results = model.predict(
    source=str(VIDEO_PATH),
    classes=TARGET_CLASSES,
    conf=0.25,
    imgsz=640,
    save=True,
    stream=True,
    project=str(ROOT / "runs"),
    name="video_detection",
    exist_ok=True,
    verbose=False,
)

frame_count = 0
detected_frames = 0

for result in results:
    frame_count += 1

    if len(result.boxes) > 0:
        detected_frames += 1

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            print(
                f"Frame {frame_count}: "
                f"{result.names[class_id]} "
                f"({confidence:.2f})"
            )

print(f"\nFrames processed: {frame_count}")
print(f"Frames with detections: {detected_frames}")
print("Output folder: runs/video_detection")
