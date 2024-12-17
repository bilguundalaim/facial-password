from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
from main import get_landmarks_from_frame, normalize_landmarks, generate_password

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_landmarks', methods=['POST'])
def capture_landmarks():
    data = request.get_json()
    frame_data = data['frame']
    frame_data = frame_data.split(',')[1]  # Remove the data URL prefix
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    # Extract landmarks and draw them on the frame
    landmarks, frame_with_landmarks = get_landmarks_from_frame(frame)
    if landmarks:
        normalized_landmarks = normalize_landmarks(landmarks)
        password = generate_password(normalized_landmarks, length=16)
        
        # Encode the frame with landmarks back to base64
        _, buffer = cv2.imencode('.png', frame_with_landmarks)
        frame_with_landmarks_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "generated_password": password,
            "frame_with_landmarks": "data:image/png;base64," + frame_with_landmarks_base64
        })
    else:
        return jsonify({"error": "No landmarks extracted."})

if __name__ == '__main__':
    app.run(debug=True)