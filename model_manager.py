import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from collections import deque
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
tf.get_logger().setLevel('ERROR')

class SignModelManager:
    """Multi-Modal Sign Language Recognition Manager."""
    
    def __init__(self):
        self.active_mode = None
        self.model = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.asl_path = './saved_models/asl_classifier.keras'
        self.isl_path = './saved_models/isl_classifier.keras'
        self.labels = {i: chr(65 + i) for i in range(26)}
        
        self.lighting_history = deque(maxlen=30)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.lighting_stable = False
        self._frame_count = 0
        
        self.prediction_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)
        self.stable_letter = "Scanning..."
        self.stable_confidence = 0.0
        self.no_hand_frames = 0
        self.cooldown_frames = 8
        
        self.asl_settings = {'min_confidence': 0.90, 'consensus_required': 0.75, 'min_frames_before_output': 12}
        self.isl_settings = {'min_confidence': 0.80, 'consensus_required': 0.70, 'min_frames_before_output': 15}
        
        self.cooldown_counter = 0
        self.last_output_letter = None
        self.repeat_ready = False
        self.frames_since_last_output = 0
        self.different_gesture_detected = False
        self.repeat_window_frames = 15

    def initialize_pipeline(self, mode):
        if self.active_mode == mode and self.hands is not None:
            return True
        self.shutdown_pipeline()
        if mode == 'ASL':
            if not os.path.exists(self.asl_path): return False
            try:
                self.model = keras.models.load_model(self.asl_path)
                dummy = np.random.randn(1, 63).astype(np.float32)
                self.model.predict(dummy, verbose=0)
                self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0)
                self.active_mode = 'ASL'
                self._reset_state()
                return True
            except Exception as e:
                print(f"Error: {e}")
                return False
        elif mode == 'ISL':
            if not os.path.exists(self.isl_path): return False
            try:
                self.model = keras.models.load_model(self.isl_path)
                dummy = np.random.randn(1, 126).astype(np.float32)
                self.model.predict(dummy, verbose=0)
                self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity=1)
                self.active_mode = 'ISL'
                self._reset_state()
                return True
            except Exception as e:
                print(f"Error: {e}")
                return False
        return False

    def _reset_state(self):
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.stable_letter = "Scanning..."
        self.stable_confidence = 0.0
        self.no_hand_frames = 0
        self.cooldown_counter = 0
        self.last_output_letter = None
        self.repeat_ready = False
        self.frames_since_last_output = 0
        self.different_gesture_detected = False
        self.lighting_history.clear()
        self.lighting_stable = False
        self._frame_count = 0

    def extract_landmark_features(self, hand_landmarks):
        features = []
        wrist = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
            features.append(lm.z - wrist.z)
        return np.array(features, dtype=np.float32)

    def is_open_palm(self, hand_landmarks, hand_label):
        lm = hand_landmarks.landmark
        thumb_open = lm[4].x < lm[3].x if hand_label == "Right" else lm[4].x > lm[3].x
        fingers = [thumb_open, lm[8].y < lm[6].y, lm[12].y < lm[10].y, lm[16].y < lm[14].y, lm[20].y < lm[18].y]
        if sum(fingers) != 5: return False
        dx, dy = abs(lm[9].x - lm[0].x), abs(lm[9].y - lm[0].y)
        if dy <= dx * 1.5: return False
        for tip in [8, 12, 16, 20]:
            if lm[tip].y > lm[0].y: return False
        return True

    def _check_hand_change(self):
        if len(self.prediction_history) < 10: return False
        recent = list(self.prediction_history)[-15:]
        recent_confs = list(self.confidence_history)[-15:]
        if len(set(recent)) > 2: return True
        if "Scanning..." in recent: return True
        if sum(1 for c in recent_confs if c < 0.60) >= 5: return True
        if self.frames_since_last_output > 25: return True
        return False

    def _get_smart_prediction(self, raw_letter, raw_confidence):
        self.prediction_history.append(raw_letter)
        self.confidence_history.append(raw_confidence)
        self.frames_since_last_output += 1
        if self.cooldown_counter > 0: self.cooldown_counter -= 1
        if self.last_output_letter is not None:
            if self._check_hand_change():
                self.repeat_ready = True
                self.different_gesture_detected = True
        settings = self.isl_settings if self.active_mode == 'ISL' else self.asl_settings
        min_frames = settings['min_frames_before_output']
        min_conf = settings['min_confidence']
        consensus_req = settings['consensus_required']
        if len(self.prediction_history) < min_frames: return "Hold steady...", 0.0
        recent_predictions = list(self.prediction_history)[-15:]
        recent_confidences = list(self.confidence_history)[-15:]
        consensus_ratio = recent_predictions.count(raw_letter) / len(recent_predictions)
        letter_confidences = [recent_confidences[i] for i, l in enumerate(recent_predictions) if l == raw_letter]
        avg_confidence = np.mean(letter_confidences) if letter_confidences else 0
        effective_min_confidence = min_conf + 0.05 if not self.lighting_stable else min_conf
        if consensus_ratio < consensus_req: return "Hold steady...", avg_confidence
        if avg_confidence < effective_min_confidence: return "Hold steady...", avg_confidence
        if raw_letter in ("Scanning...", "Hold steady..."): return raw_letter, 0.0
        if raw_letter == self.last_output_letter:
            if self.repeat_ready and self.different_gesture_detected:
                self.stable_letter = raw_letter
                self.stable_confidence = avg_confidence
                self.last_output_letter = raw_letter
                self.cooldown_counter = self.cooldown_frames
                self.repeat_ready = False
                self.different_gesture_detected = False
                self.frames_since_last_output = 0
                return raw_letter, avg_confidence
            elif self.cooldown_counter > 0:
                return "Hold steady...", avg_confidence
            else:
                return self.stable_letter, self.stable_confidence
        self.stable_letter = raw_letter
        self.stable_confidence = avg_confidence
        self.last_output_letter = raw_letter
        self.cooldown_counter = self.cooldown_frames
        self.repeat_ready = False
        self.different_gesture_detected = False
        self.frames_since_last_output = 0
        return raw_letter, avg_confidence

    def process_frame(self, frame_rgb):
        if self.hands is None or self.model is None:
            return "Scanning...", 0.0
        try:
            self._frame_count += 1
            results = self.hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                self.no_hand_frames += 1
                if self.no_hand_frames > 30: self._reset_state()
                return "Scanning...", 0.0
            self.no_hand_frames = 0
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                if self.is_open_palm(hand_landmarks, label):
                    self._reset_state()
                    return "SPACE", 1.0
            if self.active_mode == 'ASL':
                hand_landmarks = results.multi_hand_landmarks[0]
                features = self.extract_landmark_features(hand_landmarks)
                input_data = features.reshape(1, -1).astype(np.float32)
                predictions = self.model.predict(input_data, verbose=0)
                raw_letter = self.labels.get(np.argmax(predictions[0]), '?')
                raw_confidence = float(np.max(predictions[0]))
                return self._get_smart_prediction(raw_letter, raw_confidence)
            elif self.active_mode == 'ISL':
                left_features = np.zeros(63, dtype=np.float32)
                right_features = np.zeros(63, dtype=np.float32)
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    features = self.extract_landmark_features(hand_landmarks)
                    if handedness == "Left": left_features = features
                    else: right_features = features
                combined = np.concatenate([left_features, right_features])
                input_data = combined.reshape(1, -1).astype(np.float32)
                predictions = self.model.predict(input_data, verbose=0)
                raw_letter = self.labels.get(np.argmax(predictions[0]), '?')
                raw_confidence = float(np.max(predictions[0]))
                return self._get_smart_prediction(raw_letter, raw_confidence)
        except Exception as e:
            print(f"Frame processing error: {e}")
            return "Scanning...", 0.0
        return "Scanning...", 0.0

    def shutdown_pipeline(self):
        if self.hands is not None:
            try: self.hands.close()
            except: pass
        self.hands = None
        self.model = None
        self.active_mode = None
        self._reset_state()
        tf.keras.backend.clear_session()
        print("Pipeline shutdown complete.")