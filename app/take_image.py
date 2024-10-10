import dlib
import numpy as np
from sklearn.preprocessing import normalize
import cv2  # OpenCV to capture live camera feed
import os
from datetime import datetime
from app.embeddings import detect_and_extract_embedding
from app.database import insert_datad,get_all_embedding

 
# Function to capture an image from the live camera and process it
def capture_image_from_camera(user_id, user_name):
    # Initialize camera capture (0 is the default camera index)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None

    print("Press 's' to capture the image and extract the face embedding.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Press "s" to capture the image', frame)

        # Wait for 'q' key to capture the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Image captured.")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Call the face embedding function
    embedding, aligned_face = detect_and_extract_embedding(frame)

    if embedding is not None:
        print(f"Embedding for user {user_name} (ID: {user_id}) extracted successfully.")
        insert_datad(
        user_id, user_name, embedding.tolist(),
        created_at=datetime.utcnow().isoformat()  # ISO 8601 formatted timestamp
        )
    else:
        print(f"Failed to extract embedding for user {user_name} (ID: {user_id}).")


# Main function to run the program
if __name__ == "__main__":
    # Example: Ask user for input (can be replaced with real user data)
    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")
    # Capture the image from the live camera feed and extract embedding
    capture_image_from_camera(user_id, user_name)
