import cv2
import dlib
import numpy as np
import hashlib
import string
import secrets
from cryptography.fernet import Fernet

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Generate an encryption key
key = Fernet.generate_key()
cipher = Fernet(key)

def get_landmarks_from_frame(frame):
    """Extract facial landmarks from a single video frame and draw them on the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, frame  # No face detected
    
    # Use the first detected face
    landmarks = predictor(gray, faces[0])
    points = [(p.x, p.y) for p in landmarks.parts()]
    
    # Draw the landmarks on the frame
    for (x, y) in points:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    return points, frame

def normalize_landmarks(landmarks):
    """Normalize landmarks by centering and flattening."""
    landmarks = np.array(landmarks)
    mean = np.mean(landmarks, axis=0)
    normalized = landmarks - mean
    return normalized.flatten()

def generate_password(feature_vector, length=16):
    """Generate a secure password from the feature vector."""
    feature_bytes = feature_vector.tobytes()
    hash_value = hashlib.sha256(feature_bytes).hexdigest()
    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password

def encrypt_password(password):
    """Encrypt the password."""
    return cipher.encrypt(password.encode())