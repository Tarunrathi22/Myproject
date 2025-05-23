from flask import Flask, render_template, Response, request, session, redirect
import cv2
import numpy as np
import csv
import os
import atexit
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load model and emotions
model = load_model("emotion_recognition_model.h5")
emotions = ['angry', 'happy', 'sad', 'neutral', 'disgust', 'fear', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera setup
cap = cv2.VideoCapture(0)
frame_count = 0

# Logging CSV setup
log_file = "users.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Username", "Emotion", "Confidence"])

@atexit.register
def cleanup():
    cap.release()
    cv2.destroyAllWindows()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect('/detect')
    return render_template('index.html')

@app.route('/detect')
def detect():
    username = session.get('username')
    if not username:
        return redirect('/')
    return render_template('detect.html', username=username)

def gen_frames(username):
    global frame_count

    while True:
        if not cap.isOpened():
            print("Camera not opened.")
            continue

        success, frame = cap.read()
        if not success or frame is None:
            print("❌ Failed to read frame from webcam.")
            continue

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(roi, verbose=0)[0]
            max_index = np.argmax(preds)
            label = emotions[max_index]
            confidence = preds[max_index]

            frame_count += 1
            if frame_count % 10 == 0:
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        username,
                        label,
                        f"{confidence:.4f}"
                    ])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"{label.capitalize()} ({confidence*100:.1f}%)",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    username = session.get('username', 'guest')
    return Response(gen_frames(username), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    