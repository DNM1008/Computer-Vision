import cv2
from collections import deque
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    # Email configuration
    sender_email = "dungnguyen10082000@gmail.com"
    sender_password = "eixx bliz ynnn abvv"

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_video(video_path):
    # Load YOLO model and ensure it uses the GPU
    model = YOLO("yolov8n.pt")  # Use pre-trained YOLOv8 nano model
    device = "cuda"  # Force usage of GPU
    model.to(device)  # Send the model to the GPU

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Sliding window to keep track of people counts in the last 20 frames
    people_count_window = deque(maxlen=20)
    email_sent = False  # Track if email has been sent

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Perform YOLO inference on the frame
        results = model.predict(frame, device=device, half=True)  # Use half-precision (FP16) for faster inference

        # Count the number of people detected
        people_count = 0
        for result in results[0].boxes:
            if result.cls == 0:  # Class 0 corresponds to 'person' in COCO dataset
                people_count += 1

        # Update the sliding window
        people_count_window.append(people_count)

        # Display the frame with the count
        cv2.putText(frame, f"People count: {people_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("YOLO Detection", frame)

        # Check if 12 or more frames in the last 20 have at least 7 people
        if sum(1 for count in people_count_window if count >= 7) >= 12:
            print("Warning: 7 or more people detected in 12 of the last 20 frames!")

            # Send an email if not already sent
            if not email_sent:
                subject = "Warning: High People Count Detected"
                body = "7 or more people have been detected in 12 of the last 20 frames."
                to_email = "dungnguyen10082000@gmail.com"
                send_email(subject, body, to_email)
                email_sent = True

            print("Pausing video...")
            cv2.putText(frame, "WARNING: 7+ PEOPLE DETECTED", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("YOLO Detection", frame)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            people_count_window.clear()  # Clear the window after pausing

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video.mp4' with the path to your video file
process_video("video (1).mp4")
