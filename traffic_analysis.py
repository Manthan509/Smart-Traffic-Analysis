
from ultralytics import YOLO
import cv2

# 1. Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# 2. Input and output paths
image_path = "dataset/input/traffic.jpg"
output_path = "dataset/output/traffic_analysis.jpg"

# 3. Define vehicle categories
vehicle_classes = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
]

# 4. Initialize counters
vehicle_count = 0
person_count = 0

vehicle_breakdown = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0,
    "bicycle": 0
}

# 5. Run YOLO detection
results = model(image_path)

# 6. Process detected objects
for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Ignore detections below 40% confidence
        if confidence < 0.40:
            continue

        # Count vehicles
        if class_name in vehicle_classes:
            vehicle_count += 1
            vehicle_breakdown[class_name] += 1

        # Count people
        elif class_name == "person":
            person_count += 1

# 7. Calculate traffic density
if vehicle_count <= 5:
    traffic_density = "Low"

elif vehicle_count <= 12:
    traffic_density = "Medium"

else:
    traffic_density = "High"

# 8. Display results
print("\n========== SMART TRAFFIC ANALYSIS ==========")

print(f"\nTotal Vehicles: {vehicle_count}")
print(f"Total People: {person_count}")

print("\nVehicle Breakdown:")

for vehicle, count in vehicle_breakdown.items():
    print(f"{vehicle.capitalize()}: {count}")

print(f"\nTraffic Density: {traffic_density}")

# 9. Draw bounding boxes
for result in results:

    annotated_image = result.plot()

    # Add vehicle count
    cv2.putText(
        annotated_image,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Add traffic density
    cv2.putText(
        annotated_image,
        f"Density: {traffic_density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Save final image
    cv2.imwrite(output_path, annotated_image)

print(f"\nAnalysis image saved at: {output_path}")
print("\nAnalysis completed successfully!")
