
from ultralytics import YOLO
import cv2
from pathlib import Path

# 1. Load YOLO model
model = YOLO("yolov8n.pt")

# 2. Define video paths
input_video = "dataset/input/traffic.mp4"
output_video = "dataset/output/analysed_traffic.mp4"

# 3. Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 4. Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

# 5. Create output video
Path("dataset/output").mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# 6. Vehicle classes
vehicle_classes = [
    "car", "bus", "truck", "motorcycle", "bicycle"
]

frame_count = 0
total_vehicle_detections = 0

print("Starting video analysis...")

# 7. Process video frame by frame
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO on current frame
    result = model(frame, conf=0.40, verbose=False)[0]

    vehicle_count = 0
    person_count = 0

    # Count detected objects
    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in vehicle_classes:
            vehicle_count += 1

        elif class_name == "person":
            person_count += 1

    # Calculate density for this frame
    if vehicle_count <= 5:
        density = "Low"
    elif vehicle_count <= 12:
        density = "Medium"
    else:
        density = "High"

    total_vehicle_detections += vehicle_count
    frame_count += 1

    # Draw bounding boxes
    annotated_frame = result.plot()

    # Add traffic information
    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Density: {density}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Save processed frame
    out.write(annotated_frame)

    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames")

# 8. Release resources
cap.release()
out.release()

# 9. Display final results
if frame_count > 0:

    average_vehicles = total_vehicle_detections / frame_count

    print("\n===== VIDEO TRAFFIC ANALYSIS =====")
    print(f"Total frames processed: {frame_count}")
    print(f"Average vehicles per frame: {average_vehicles:.2f}")

    print(f"Output video saved at: {output_video}")

    print("Video analysis completed!")

else:
    print("No video frames were processed.")
