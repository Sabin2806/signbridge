import os
import sys
import base64

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
latest_letter = "Scanning..."
latest_confidence = 0.0

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

# ============================================
# BROWSER WEBCAM PREDICTION ENDPOINT
# ============================================

@app.route('/api/predict_frame', methods=['POST'])
def api_predict_frame():
    """Process frame from browser webcam with landmarks."""
    global latest_letter, latest_confidence
    try:
        data = request.get_json()
        frame_data = data.get('frame', '')
        
        # Decode base64 image
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'letter': 'Scanning...', 'confidence': 0.0, 'annotated_frame': None})
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        annotated_frame = frame.copy()
        
        if manager.hands is not None and manager.model is not None:
            # Process frame for prediction
            latest_letter, latest_confidence = manager.process_frame(img_rgb)
            
            # Draw landmarks
            try:
                results = manager.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 243, 255), thickness=3, circle_radius=5),
                            mp_drawing.DrawingSpec(color=(157, 78, 221), thickness=2)
                        )
            except Exception as e:
                print(f"Landmark drawing error: {e}")
        else:
            latest_letter = "Pipeline not ready..."
            latest_confidence = 0.0
        
        # Encode annotated frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_data = f'data:image/jpeg;base64,{annotated_base64}'
        
        return jsonify({
            'letter': latest_letter,
            'confidence': float(latest_confidence),
            'annotated_frame': annotated_data
        })
    
    except Exception as e:
        print(f"Error in predict_frame: {e}")
        return jsonify({'letter': 'Scanning...', 'confidence': 0.0, 'annotated_frame': None})

# ============================================
# LEGACY ENDPOINTS (Work locally)
# ============================================

@app.route('/api/predict')
def api_predict():
    global latest_letter, latest_confidence
    return jsonify({'letter': latest_letter, 'confidence': float(latest_confidence)})

@app.route('/api/terminate', methods=['POST'])
def api_terminate():
    print("\nPipeline termination requested...")
    manager.shutdown_pipeline()
    return jsonify({'status': 'Pipeline terminated.', 'redirect': '/selection'})

@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    print("\nFull shutdown requested...")
    manager.shutdown_pipeline()
    response = jsonify({'status': 'System shutting down.', 'message': 'You can close this window.'})
    threading.Timer(1.5, shutdown_server).start()
    return response

def shutdown_server():
    print("\nServer shutdown initiated...")
    manager.shutdown_pipeline()
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("SignBridge - Multi-Modal Sign Recognition System")
    print("="*50)
    
    port = int(os.environ.get('PORT', 10000))
    print(f"Access: http://0.0.0.0:{port}")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        manager.shutdown_pipeline()
        print("\nSignBridge shutdown complete.")