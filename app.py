from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from model_manager import SignModelManager
import threading
import os
import signal
import base64

app = Flask(__name__)
manager = SignModelManager()

camera = None
latest_letter = "Scanning..."
latest_confidence = 0.0

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

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

@app.route('/api/predict_frame', methods=['POST'])
def api_predict_frame():
    """
    Receives base64 frame from frontend,
    processes through MediaPipe + CNN,
    returns prediction with annotated frame.
    """
    global latest_letter, latest_confidence
    
    try:
        data = request.get_json()
        
        if data is None or 'frame' not in data:
            return jsonify({
                'letter': 'Scanning...',
                'confidence': 0.0,
                'annotated_frame': None
            })
        
        frame_data = data['frame']
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'letter': 'Scanning...',
                'confidence': 0.0,
                'annotated_frame': None
            })
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if manager.hands is not None and manager.model is not None:
            latest_letter, latest_confidence = manager.process_frame(img_rgb)
        else:
            latest_letter = "Scanning..."
            latest_confidence = 0.0
        
        annotated_frame = None
        
        if manager.hands is not None:
            try:
                import mediapipe as mp
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                
                results = manager.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 243, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(17, 17, 17), thickness=2)
                        )
                
                _, buffer = cv2.imencode('.jpg', frame)
                annotated_frame = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"Drawing error: {e}")
        
        return jsonify({
            'letter': latest_letter,
            'confidence': float(latest_confidence),
            'annotated_frame': annotated_frame
        })
        
    except Exception as e:
        print(f"Predict frame error: {e}")
        return jsonify({
            'letter': 'Scanning...',
            'confidence': 0.0,
            'annotated_frame': None
        })

@app.route('/api/predict', methods=['GET'])
def api_predict():
    global latest_letter, latest_confidence
    return jsonify({
        'letter': latest_letter,
        'confidence': float(latest_confidence)
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = get_camera()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/terminate', methods=['POST'])
def api_terminate():
    manager.shutdown_pipeline()
    release_camera()
    return jsonify({'status': 'Pipeline terminated.', 'redirect': '/selection'})

@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    manager.shutdown_pipeline()
    release_camera()
    return jsonify({'status': 'System shutting down.'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("SignBridge - Multi-Modal Sign Recognition System")
    print("="*50)
    print("Access: http://localhost:5000")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        release_camera()
        manager.shutdown_pipeline()
        print("SignBridge shutdown complete.")