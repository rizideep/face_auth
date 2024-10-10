import psycopg2
import numpy as np
from pymongo import MongoClient
import pymongo 
 
def get_all_embedding_static():
    uri = f"mongodb+srv://new_user1:deep617@cluster1.i8ag3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
    # Create a new client and connect to the server
    client = MongoClient(uri)
    # Access the 'face_auth' database
    db = client['face_auth']
    # Access the 'face_db_collection' collection
    collection = db['face_db_collection']
    # Query to retrieve all embeddings
    # Query to retrieve all data from all documents (no field restrictions)
    embeddings = collection.find({})  # This will retrieve all fields for all documents
    # Convert cursor to list
    embedding_list = list(embeddings)
    return embedding_list

def insert_datad_static(user_id, name, embedding, created_at):
    uri = "mongodb+srv://new_user1:deep617@cluster1.i8ag3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
     # Create a new client and connect to the server
    client = MongoClient(uri)
     # Access the 'face_auth' database
    db = client['face_auth']
    # Access the 'face_db_collection' collection
    collection = db['face_db_collection']
    document = {
        "user_id": user_id,
        "name": name,
        "embedding": embedding,
        "created_at": created_at
    }
    result = collection.insert_one(document)
    collection.create_index([("user_id", pymongo.ASCENDING)])
    print(f"Inserted document ID: {result.inserted_id}")
    # List all collections in the 'face_auth' database
    collections = db.list_collection_names()
    print("Collections:", collections)
    # # Retrieve and print all documents from the 'face_db_collection'
    # for doc in collection.find():
    #  print(doc) 