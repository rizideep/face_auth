import dlib
import numpy as np
from sklearn.preprocessing import normalize
import cv2  # OpenCV to handle image processing
import os
from datetime import datetime

from database_static import insert_datad_static
from embeddings_static import detect_and_extract_embedding_static 


# Function to load a static image, process it, and extract face embedding
def process_image_from_file(image_path, user_id, user_name):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return None

    # Load the image
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load the image from '{image_path}'.")
        return None

    # # Display the image (optional)
    # cv2.imshow(f"Loaded Image for {user_name} (ID: {user_id})", frame)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()
    # Call the face embedding function
    embedding, aligned_face = detect_and_extract_embedding_static(frame)
    if embedding is not None:
        insert_datad_static(
            user_id, user_name, embedding.tolist(),
            created_at=datetime.utcnow().isoformat()  # ISO 8601 formatted timestamp
        )
    else:
        print(f"Failed to extract embedding for user {user_name} (ID: {user_id}).")



# Main function to run the program
if __name__ == "__main__":
    # Example: Ask user for input (can be replaced with real user data)
    # user_id = input("Enter User ID: ")
    # user_name = input("Enter User Name: ")
    
    user_id = "10"
    user_name = "shikha"
    # Static image path (replace this with your own image file path)
    image_path =  "user/shikha.jpg"

    # Process the image from the file and extract embedding
    process_image_from_file(image_path, user_id, user_name)
