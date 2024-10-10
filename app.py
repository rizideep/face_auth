from flask import Flask, request, jsonify
from static.embeddings_static import detect_and_extract_embedding_static
from PIL import Image
import io
import numpy as np
from datetime import datetime
import faiss
from static.database_static import get_all_embedding_static, insert_datad_static
from static.authnew import authenticate_face_static

app = Flask(__name__)

# Build the FAISS index
def build_faiss_index():
    embeddings = get_all_embedding_static()  # Convert Cursor to list if necessary
    embedding_vectors = []
    metadata = []

    for emb in embeddings:
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

# Initialize FAISS index and embeddings
faiss_index, all_embeddings = build_faiss_index()

@app.route('/register/', methods=['POST'])
def register_user():
    # Get the required fields from form data
    user_id = request.form.get("user_id")
    user_name = request.form.get("name")
    image_file = request.files.get('image')

    # Check if all required fields are provided
    if not user_id or not user_name or not image_file:
        return jsonify({"status": "error", "message": "user_id, name, and image are required"}), 400

    # Check if the user is already registered
    if user_id in [emb[1] for emb in all_embeddings]:  # Assuming the user_id is in the metadata list
        return jsonify({"status": "error", "message": "User already registered"}), 400

    # Load the uploaded image and convert to numpy array
    try:
        img = Image.open(io.BytesIO(image_file.read()))
        np_img = np.array(img)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing image: {str(e)}"}), 400

    # Extract embedding from the image
    embedding, aligned_face = detect_and_extract_embedding_static(np_img)

    if embedding is None:
        return jsonify({"status": "error", "message": "No face detected in the image"}), 400

    # Insert the new user's embedding
    insert_datad_static(
        user_id=user_id, 
        name=user_name, 
        embedding=embedding.tolist(),
        created_at=datetime.utcnow().isoformat()  # ISO 8601 formatted timestamp
    )

    return jsonify({"status": "success", "message": "User registered successfully", "user_id": user_id, "name": user_name}), 200


@app.route("/authenticate/", methods=['POST'])
def authenticate():
    # Load the uploaded image and convert to numpy array
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"status": "error", "message": "Image file is required"}), 400
    
    img = Image.open(io.BytesIO(image_file.read()))
    np_img = np.array(img)

    # Extract embedding from the image
    embedding, aligned_face = detect_and_extract_embedding_static(np_img)

    if embedding is None:
        return jsonify({"status": "error", "message": "No face detected"})

    # Authenticate the face using the embedding
    authenticated, _id, user_id, user_name = authenticate_face_static(embedding)

    if authenticated:
        return jsonify({"status": "success", "user_id": user_id, "name": user_name})
    else:
        return jsonify({"status": "error", "message": "Authentication failed"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
 