import faiss
import numpy as np
import dlib
from sklearn.preprocessing import normalize
from .database_static import get_all_embedding_static



# Load dlib models (Ensure that the .dat files are correctly downloaded and placed)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Build the FAISS index
def build_faiss_index():
    embeddings =  get_all_embedding_static()  # Convert Cursor to list if necessary
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
     #  print(f"emp:  {emb}")  # Store metadata

    # Convert lists to numpy arrays
    embedding_vectors = np.array(embedding_vectors)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_vectors.shape[1])
    index.add(embedding_vectors)
    
    return index, metadata

faiss_index, all_embeddings = build_faiss_index()

def authenticate_face_static(input_embedding, threshold=0.6):
    if faiss_index is None:
        return False, None, None, None
    
    # distances, indices = faiss_index.search(np.array([input_embedding]), 1)
    # best_distance = distances[0][0]
    # best_index = indices[0][0]
    
    # if best_distance < threshold:
    #     _id, user_id, user_name = all_embeddings[best_index]
    #     print(f"Matched User ID: {user_id}, Name: {user_name}, ID: {_id}")
    #     return True, _id, user_id, user_name
    # else:
    #     return False, None, None, None
    

    # Perform a search on the FAISS index
    distances, indices = faiss_index.search(np.array([input_embedding]), 1)
    best_distance = distances[0][0]
    best_index = indices[0][0]

    # Check for an exact match (distance == 0)
    if best_distance <  0.1:  # Exact match (may consider a very small tolerance if needed)
        _id, user_id, user_name = all_embeddings[best_index]
        print(f"Exact match found! User ID: {user_id}, Name: {user_name}, ID: {_id}")
        return True, _id, user_id, user_name
    else:
        print("No exact match found.")
        return False, None, None, None

 
 