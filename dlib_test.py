import faiss
import numpy as np
import cv2
import dlib
from sklearn.preprocessing import normalize
from app.database import get_all_embedding
from app.embeddings import detect_and_extract_embedding

# Load dlib models (Ensure that the .dat files are correctly downloaded and placed)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Build the FAISS index
def build_faiss_index():
    embeddings = get_all_embedding()  # Convert Cursor to list if necessary
    embedding_vectors = []
    metadata = []

    for emb in embeddings:
        # Extract values from each document (which is likely a dictionary)
        _id = emb.get('_id')
        user_id = emb.get('user_id')
        name = emb.get('name')
        embedding = emb.get('embedding')  # Assuming this is the embedding array
        embedding_vectors.append(embedding)
        metadata.append((_id, user_id, name))

    # Convert lists to numpy arrays
    embedding_vectors = np.array(embedding_vectors)

    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_vectors.shape[1])
    index.add(embedding_vectors)

    return index, metadata

faiss_index, all_embeddings = build_faiss_index()

def authenticate_face(input_embedding, threshold=0.6):
    if faiss_index is None:
        print("Error: FAISS index is not initialized.")
        return False, None, None, None

    # Handle cases where embedding might be None
    if input_embedding is None:
        print("Error: No valid embedding found for the face.")
        return False, None, None, None

    # Perform FAISS search
    distances, indices = faiss_index.search(np.array([input_embedding]), 1)
    best_distance = distances[0][0]
    best_index = indices[0][0]

    if best_distance < threshold:
        _id, user_id, user_name = all_embeddings[best_index]
        print(f"Matched User ID: {user_id}, Name: {user_name}, ID: {_id}")
        return True, _id, user_id, user_name
    else:
        return False, None, None, None

# Load the static image instead of capturing video
image_path = "user/sunnyt.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Image not found or path is incorrect.")
else:
    # Load the image and extract embedding
    embedding, aligned_face = detect_and_extract_embedding(frame)

    if embedding is not None:
        authenticated, _id, user_id, user_name = authenticate_face(embedding)

        # Display result
        if authenticated:
            cv2.putText(frame, f"User ID: {user_id}, Name: {user_name}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the image with a rectangle and result
        window_name = "Face with Rectangle"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a named window
        cv2.resizeWindow(window_name, 800, 800)  # Resize window to 800x600
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
    else:
        print("Error: Failed to extract embedding from the image.")
 