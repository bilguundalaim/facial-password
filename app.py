from flask import Flask, render_template, jsonify, request, redirect, url_for
import cv2
import numpy as np
import base64
from main import get_landmarks_from_frame, normalize_landmarks, generate_password
from database import init_db, add_user, get_user

app = Flask(__name__)
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        add_user(username, password)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and user[1] == password:
            return f"Login successful for username: {username}"
        else:
            return "Invalid username or password. Please try again."
    return render_template('login.html')

@app.route('/capture_landmarks', methods=['POST'])
def capture_landmarks():
    data = request.get_json()
    frame_data = data['frame']
    frame_data = frame_data.split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    landmarks, frame_with_landmarks = get_landmarks_from_frame(frame)
    if landmarks:
        normalized_landmarks = normalize_landmarks(landmarks)
        password = generate_password(normalized_landmarks, length=16)
        
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