import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from collections import deque
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class SignModelManager:
    def __init__(self):
        self.active_mode = None
        self.model = None
        self.mp_hands = mp.solutions.hands
        self.hands = None

        self.asl_path = './saved_models/asl_classifier.keras'
        self.isl_path = './saved_models/isl_classifier.keras'
        self.labels = {i: chr(65 + i) for i in range(26)}
        
        self.prediction_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=20)
        self.min_confidence = 0.85
        self.consensus_required = 0.70
        self.min_frames_before_output = 10
        self.cooldown_frames = 6
        self.cooldown_counter = 0
        self.last_output_letter = None
        self.stable_letter = "Scanning..."
        self.stable_confidence = 0.0
        self.no_hand_frames = 0

    def initialize_pipeline(self, mode):
        if self.active_mode == mode and self.hands is not None:
            print(f"{mode} pipeline already active.")
            return True

        self.shutdown_pipeline()

        if mode == 'ASL':
            if not os.path.exists(self.asl_path):
                print(f"ERROR: Model not found at {self.asl_path}")
                return False
            try:
                self.model = keras.models.load_model(self.asl_path)
                print(f"ASL Model loaded.")
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.active_mode = 'ASL'
                self._reset_state()
                print("ASL Pipeline Initialized.")
                return True
            except Exception as e:
                print(f"ERROR: {e}")
                return False

        elif mode == 'ISL':
            if not os.path.exists(self.isl_path):
                print(f"ERROR: Model not found at {self.isl_path}")
                return False
            try:
                self.model = keras.models.load_model(self.isl_path)
                print(f"ISL Model loaded.")
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.active_mode = 'ISL'
                self._reset_state()
                print("ISL Pipeline Initialized.")
                return True
            except Exception as e:
                print(f"ERROR: {e}")
                return False

        return False

    def _reset_state(self):
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.cooldown_counter = 0
        self.last_output_letter = None
        self.stable_letter = "Scanning..."
        self.stable_confidence = 0.0
        self.no_hand_frames = 0

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
        if hand_label == "Right":
            thumb_open = lm[4].x < lm[3].x
        else:
            thumb_open = lm[4].x > lm[3].x
        fingers = [thumb_open, lm[8].y < lm[6].y, lm[12].y < lm[10].y,
                   lm[16].y < lm[14].y, lm[20].y < lm[18].y]
        if sum(fingers) != 5:
            return False
        dx = abs(lm[9].x - lm[0].x)
        dy = abs(lm[9].y - lm[0].y)
        return dy > dx * 1.5

    def process_frame(self, frame_rgb):
        if self.hands is None or self.model is None:
            return "Scanning...", 0.0

        try:
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                self.no_hand_frames += 1
                if self.no_hand_frames > 30:
                    self._reset_state()
                return "Scanning...", 0.0

            self.no_hand_frames = 0

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                if self.is_open_palm(hand_landmarks, label):
                    return "SPACE", 1.0

            if self.active_mode == 'ASL':
                hand_landmarks = results.multi_hand_landmarks[0]
                features = self.extract_landmark_features(hand_landmarks)
                input_data = features.reshape(1, -1).astype(np.float32)
            elif self.active_mode == 'ISL':
                left_f = np.zeros(63, dtype=np.float32)
                right_f = np.zeros(63, dtype=np.float32)
                for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    feats = self.extract_landmark_features(hand_lm)
                    if handedness == "Left":
                        left_f = feats
                    else:
                        right_f = feats
                input_data = np.concatenate([left_f, right_f]).reshape(1, -1).astype(np.float32)
            else:
                return "Scanning...", 0.0

            predictions = self.model.predict(input_data, verbose=0)
            raw_letter = self.labels.get(np.argmax(predictions[0]), '?')
            raw_confidence = float(np.max(predictions[0]))

            # Simple smart filter
            self.prediction_history.append(raw_letter)
            self.confidence_history.append(raw_confidence)

            if len(self.prediction_history) < self.min_frames_before_output:
                return "Hold steady...", 0.0

            recent = list(self.prediction_history)[-15:]
            consensus = recent.count(raw_letter) / len(recent)

            if consensus < self.consensus_required:
                return "Hold steady...", 0.0

            if raw_confidence < self.min_confidence:
                return "Hold steady...", 0.0

            return raw_letter, raw_confidence

        except Exception as e:
            print(f"Frame processing error: {e}")
            return "Scanning...", 0.0

    def shutdown_pipeline(self):
        if self.hands is not None:
            try:
                self.hands.close()
            except:
                pass
        self.hands = None
        self.model = None
        self.active_mode = None
        self._reset_state()
        tf.keras.backend.clear_session()
        print("Pipeline shutdown complete.")