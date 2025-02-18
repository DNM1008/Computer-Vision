from ultralytics import YOLO
import cv2

# Specify the model and image path
model_name = "yolov8l"
model_path = "models/yolov8l.pt"  # Path to your model
image_path = "image.jpg"  # Path to your image
output_path = "output_image.jpg"  # Path to save the output image

def count_people_and_save_image(model_name, model_path, image_path, output_path):
    print(f"Counting people in {image_path} using {model_name}...")

    # Load the model
    model = YOLO(model_path).to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return

    # Perform inference
    results = model(image)

    # Count people (class id 0 for "person") and draw bounding boxes
    people_count = 0
    detections = results[0].boxes  # For YOLOv8, results[0].boxes gives a Box object
    for det in detections:
        if det.cls == 0:  # Check if the detected object is a person
            people_count += 1
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().flatten())
            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add a label
            cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result image
    cv2.imwrite(output_path, image)
    print(f"Total number of people detected: {people_count}")
    print(f"Output image saved to {output_path}")

    return people_count

if __name__ == "__main__":
    count_people_and_save_image(model_name, model_path, image_path, output_path)
