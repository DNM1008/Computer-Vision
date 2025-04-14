from ultralytics import YOLO

model = YOLO("../conf/small_cassette.pt")

# Print class names with indices
for class_id, class_name in model.model.names.items():
    print(f"Class ID: {class_id}, Class Name: {class_name}")
