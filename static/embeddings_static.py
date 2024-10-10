import dlib
import numpy as np
from sklearn.preprocessing import normalize
import cv2  # OpenCV to capture live camera feed

# Load dlib models (Ensure that the .dat files are correctly downloaded and placed)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


# Function to detect and extract face embeddings and show the face with a rectangle
def detect_and_extract_embedding_static(image):
    try:
        # Convert image to grayscale using OpenCV (cv2), not dlib
        print("Converting image to grayscale...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Use cv2.cvtColor here
        print("Image successfully converted to grayscale.")
        
        # Detect faces
        print("Detecting faces...")
        faces = detector(gray)
        print(f"Number of faces detected: {len(faces)}")
        
        if len(faces) == 0:
            print("No faces detected.")
            return None, None  # No face detected

        # Take the first detected face
        face = faces[0]
        print(f"Face detected at position: {face}")
        
        # Draw a rectangle around the detected face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle
        
        # Get facial landmarks
        print("Detecting facial landmarks...")
        landmarks = predictor(gray, face)
        print("Facial landmarks detected.")
        
        # Align the face
        print("Aligning face using landmarks...")
        aligned_face = dlib.get_face_chip(image, landmarks)
        print("Face alignment completed.")
        
        # Extract the face embedding
        print("Extracting face embedding...")
        embedding = np.array(face_rec_model.compute_face_descriptor(aligned_face))
        
        print("Embedding extracted.")
        
        # Normalize the embedding
        print("Normalizing the embedding...")
        embedding = normalize(embedding.reshape(1, -1))[0]
        print("*********************")
        print(f"Embedding normalized. {embedding}")
        return embedding, aligned_face
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
 