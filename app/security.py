from Crypto.Cipher import AES
import base64
import numpy as np

SECRET_KEY = b'Sixteen byte key'  # Example key (must be kept secret!)

def encrypt_embedding(embedding):
    cipher = AES.new(SECRET_KEY, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(embedding.tobytes())
    return base64.b64encode(ciphertext).decode('utf-8')

def decrypt_embedding(ciphertext):
    decoded = base64.b64decode(ciphertext)
    cipher = AES.new(SECRET_KEY, AES.MODE_EAX)
    embedding = cipher.decrypt(decoded)
    return np.frombuffer(embedding, dtype=np.float64)
