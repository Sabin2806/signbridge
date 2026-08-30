import os
import sys

# Suppress all unnecessary logging for performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from model_manager import SignModelManager
import threading
import signal

app = Flask(__name__)
manager = SignModelManager()

# Global State Management
camera = None
latest_letter = "Scanning..."
latest_confidence = 0.0

def get_camera():
    """Get or create camera instance."""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    """Release camera resources."""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def shutdown_server():
    """Shutdown the Flask server gracefully."""
    print("\n" + "="*50)
    print("🛑 Server shutdown initiated...")
    release_camera()
    manager.shutdown_pipeline()
    print("Server terminating...")
    print("="*50)
    os.kill(os.getpid(), signal.SIGINT)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/selection')
def selection():
    return render_template('selection.html')

@app.route('/recognition')
def recognition():
    mode = request.args.get('mode', 'ASL')
    success = manager.initialize_pipeline(mode)
    if not success:
        return f"Error initializing {mode} pipeline.", 500
    return render_template('recognition.html', mode=mode)

def generate_unified_stream():
    global latest_letter, latest_confidence
    cap = get_camera()
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if manager.hands is not None and manager.model is not None:
            latest_letter, latest_confidence = manager.process_frame(img_rgb)
        else:
            latest_letter = "Scanning..."
            latest_confidence = 0.0
        
        if manager.hands is not None:
            try:
                results = manager.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 243, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(17, 17, 17), thickness=2)
                        )
            except Exception:
                pass
                    
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_unified_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/predict')
def api_predict():
    global latest_letter, latest_confidence
    return jsonify({'letter': latest_letter, 'confidence': float(latest_confidence)})

@app.route('/api/terminate', methods=['POST'])
def api_terminate():
    print("\n🛑 Pipeline termination requested...")
    manager.shutdown_pipeline()
    release_camera()
    return jsonify({'status': 'Pipeline terminated.', 'redirect': '/selection'})

@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    print("\n🛑 Full shutdown requested...")
    manager.shutdown_pipeline()
    release_camera()
    response = jsonify({'status': 'System shutting down.', 'message': 'You can close this window.'})
    threading.Timer(1.5, shutdown_server).start()
    return response

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SignBridge - Multi-Modal Sign Recognition System")
    print("="*50)
    
    port = int(os.environ.get('PORT', 10000))
    print(f"Access: http://0.0.0.0:{port}")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt received.")
    finally:
        release_camera()
        manager.shutdown_pipeline()
        print("\n✅ SignBridge shutdown complete.")