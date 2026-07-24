"""
=============================================================================
SIGN LANGUAGE CNN TRAINING PIPELINE - SMART RESUME
- Checks existing models and skips completed training
- Fixes encoding error in report generation
=============================================================================
"""

import os
import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import random
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Global configuration."""
    
    ASL_DATASET_PATH = 'D:/ASL MODEL/asl dataset'
    ISL_DATASET_PATH = 'D:/ISL MODEL/isl dataset'
    SAVE_DIR = './saved_models'
    LOG_DIR = './logs'
    PLOT_DIR = './plots'
    REPORT_DIR = './reports'
    
    MAX_IMAGES_PER_LETTER = 600
    RANDOM_SEED = 42
    NUM_CLASSES = 26
    ASL_INPUT_SHAPE = (63,)
    ISL_INPUT_SHAPE = (126,)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    USE_AUGMENTATION = True
    AUGMENTATION_FACTOR = 2
    
    def __init__(self):
        for directory in [self.SAVE_DIR, self.LOG_DIR, self.PLOT_DIR, self.REPORT_DIR]:
            os.makedirs(directory, exist_ok=True)


# =============================================================================
# STATUS CHECKER
# =============================================================================

class TrainingStatus:
    """Checks what has been completed and what needs to be done."""
    
    @staticmethod
    def check_model_status(mode):
        """Check if model training is complete."""
        model_path = f'{Config.SAVE_DIR}/{mode}_classifier.keras'
        h5_path = f'{Config.SAVE_DIR}/{mode}_classifier.h5'
        checkpoint_path = f'{Config.SAVE_DIR}/{mode}_checkpoint.keras'
        
        if os.path.exists(model_path) or os.path.exists(h5_path):
            return 'complete'
        elif os.path.exists(checkpoint_path):
            return 'partial'
        else:
            return 'not_started'
    
    @staticmethod
    def check_report_status(mode):
        """Check if report was generated."""
        report_path = f'{Config.REPORT_DIR}/{mode}_training_report.txt'
        results_path = f'{Config.REPORT_DIR}/{mode}_results.json'
        
        if os.path.exists(results_path):
            return 'complete'
        elif os.path.exists(report_path):
            return 'partial'
        else:
            return 'not_started'
    
    @staticmethod
    def print_status(mode):
        """Print current status for a mode."""
        model_status = TrainingStatus.check_model_status(mode)
        report_status = TrainingStatus.check_report_status(mode)
        
        print(f"\n{mode.upper()} STATUS:")
        print(f"  Model:  {'✅ Complete' if model_status == 'complete' else '⚠️ Partial' if model_status == 'partial' else '❌ Not started'}")
        print(f"  Report: {'✅ Complete' if report_status == 'complete' else '⚠️ Partial' if report_status == 'partial' else '❌ Not started'}")
        
        return model_status, report_status


# =============================================================================
# DATASET PREPARATION (Same as before)
# =============================================================================

class DatasetPreparator:
    """Prepares dataset with trimming and landmark extraction."""
    
    def __init__(self, dataset_path, mode='asl', max_images=600):
        self.dataset_path = dataset_path
        self.mode = mode.lower()
        self.max_images = max_images
        self.mp_hands = mp.solutions.hands
        
        if self.mode == 'asl':
            self.max_hands = 1
            self.features_per_sample = 63
        else:
            self.max_hands = 2
            self.features_per_sample = 126
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([chr(65 + i) for i in range(26)])
        
        self.stats = {
            'mode': mode,
            'max_images_per_letter': max_images,
            'total_original_images': 0,
            'total_after_trimming': 0,
            'total_after_extraction': 0,
            'total_after_augmentation': 0,
            'failed_extractions': 0,
            'per_class': defaultdict(lambda: {
                'original': 0, 'after_trim': 0,
                'extracted': 0, 'failed': 0
            })
        }
    
    def trim_dataset(self):
        """Trim dataset to max_images per letter."""
        print(f"\n{'='*70}")
        print(f"DATASET TRIMMING - {self.mode.upper()}")
        print(f"{'='*70}")
        print(f"Target: {self.max_images} images per letter")
        
        letter_files = {}
        
        for letter in sorted(os.listdir(self.dataset_path)):
            letter_path = os.path.join(self.dataset_path, letter)
            
            if not os.path.isdir(letter_path) or len(letter) != 1 or not letter.isalpha():
                continue
            
            image_files = [f for f in os.listdir(letter_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            self.stats['per_class'][letter]['original'] = len(image_files)
            self.stats['total_original_images'] += len(image_files)
            
            if len(image_files) > self.max_images:
                random.seed(Config.RANDOM_SEED)
                selected_files = random.sample(image_files, self.max_images)
                self.stats['per_class'][letter]['after_trim'] = self.max_images
                print(f"  {letter}: {len(image_files)} -> {self.max_images} (trimmed)")
            elif len(image_files) == self.max_images:
                selected_files = image_files
                self.stats['per_class'][letter]['after_trim'] = self.max_images
                print(f"  {letter}: {len(image_files)} (perfect)")
            else:
                selected_files = image_files
                self.stats['per_class'][letter]['after_trim'] = len(image_files)
                print(f"  {letter}: {len(image_files)} (less than target)")
            
            letter_files[letter] = {'path': letter_path, 'files': selected_files}
        
        self.stats['total_after_trimming'] = sum(
            self.stats['per_class'][l]['after_trim'] for l in self.stats['per_class']
        )
        
        print(f"\nSummary: {self.stats['total_original_images']} -> {self.stats['total_after_trimming']}")
        
        return letter_files
    
    def extract_landmarks(self, image_path):
        """Extract MediaPipe hand landmarks."""
        image = cv2.imread(image_path)
        if image is None:
            return None, False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=self.max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            results = hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                return None, False
            
            if self.mode == 'asl':
                return self._extract_single_hand(results), True
            else:
                return self._extract_dual_hand(results), True
    
    def _extract_single_hand(self, results):
        hand_landmarks = results.multi_hand_landmarks[0]
        return self._landmarks_to_features(hand_landmarks)
    
    def _extract_dual_hand(self, results):
        left_features = np.zeros(63, dtype=np.float32)
        right_features = np.zeros(63, dtype=np.float32)
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            features = self._landmarks_to_features(hand_landmarks)
            
            if handedness == "Left":
                left_features = features
            else:
                right_features = features
        
        return np.concatenate([left_features, right_features])
    
    def _landmarks_to_features(self, hand_landmarks):
        features = []
        wrist = hand_landmarks.landmark[0]
        
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
            features.append(lm.z - wrist.z)
        
        return np.array(features, dtype=np.float32)
    
    def augment_landmarks(self, features, num_augmentations=2):
        augmented = []
        
        for _ in range(num_augmentations):
            aug_features = features.copy()
            num_landmarks = len(aug_features) // 3
            landmarks = aug_features.reshape(num_landmarks, 3)
            
            translation = np.random.normal(0, 0.02, (1, 3))
            landmarks += translation
            
            scale = np.random.uniform(0.9, 1.1)
            landmarks *= scale
            
            noise = np.random.normal(0, 0.005, landmarks.shape)
            landmarks += noise
            
            augmented.append(landmarks.reshape(-1))
        
        return augmented
    
    def process_dataset(self):
        """Process dataset and extract features."""
        letter_files = self.trim_dataset()
        
        print(f"\n{'='*70}")
        print(f"EXTRACTING LANDMARKS - {self.mode.upper()}")
        print(f"{'='*70}")
        
        X, y = [], []
        
        for letter, data in letter_files.items():
            print(f"\nProcessing '{letter}' ({len(data['files'])} images)...")
            
            for img_file in tqdm(data['files'], desc=f"  {letter}", unit='img'):
                img_path = os.path.join(data['path'], img_file)
                features, success = self.extract_landmarks(img_path)
                
                if success:
                    X.append(features)
                    y.append(letter)
                    self.stats['per_class'][letter]['extracted'] += 1
                    self.stats['total_after_extraction'] += 1
                    
                    if Config.USE_AUGMENTATION:
                        augmented = self.augment_landmarks(features, Config.AUGMENTATION_FACTOR)
                        for aug_features in augmented:
                            X.append(aug_features)
                            y.append(letter)
                            self.stats['total_after_augmentation'] += 1
                else:
                    self.stats['per_class'][letter]['failed'] += 1
                    self.stats['failed_extractions'] += 1
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        if Config.USE_AUGMENTATION:
            self.stats['total_after_augmentation'] += self.stats['total_after_extraction']
        
        self._print_stats()
        return X, y
    
    def _print_stats(self):
        print(f"\n{'='*70}")
        print(f"DATASET STATISTICS - {self.mode.upper()}")
        print(f"{'='*70}")
        
        print(f"\nOverall:")
        print(f"  Original:       {self.stats['total_original_images']}")
        print(f"  After trim:     {self.stats['total_after_trimming']}")
        print(f"  Extracted:      {self.stats['total_after_extraction']}")
        print(f"  Failed:         {self.stats['failed_extractions']}")
        
        if Config.USE_AUGMENTATION:
            print(f"  With augment:   {self.stats['total_after_augmentation']}")
        
        print(f"\nPer-Class:")
        print(f"{'Letter':<8} {'Orig':<8} {'Trim':<8} {'Extr':<8} {'Fail':<8} {'Rate':<8}")
        print("-" * 48)
        
        for letter in sorted(self.stats['per_class'].keys()):
            stats = self.stats['per_class'][letter]
            rate = (stats['extracted'] / stats['after_trim'] * 100) if stats['after_trim'] > 0 else 0
            print(f"{letter:<8} {stats['original']:<8} {stats['after_trim']:<8} "
                  f"{stats['extracted']:<8} {stats['failed']:<8} {rate:.1f}%")
    
    def save_stats(self):
        stats_path = f'{Config.REPORT_DIR}/{self.mode}_dataset_stats.json'
        stats_copy = self.stats.copy()
        stats_copy['per_class'] = dict(stats_copy['per_class'])
        
        with open(stats_path, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        print(f"Stats saved: {stats_path}")


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ModelArchitectures:
    """CNN model architectures."""
    
    @staticmethod
    def create_asl_model(input_shape=(63,), num_classes=26):
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((21, 3, 1)),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.3),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.3),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_isl_model(input_shape=(126,), num_classes=26):
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((42, 3, 1)),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.3),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.3),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.3),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        return model


# =============================================================================
# TRAINER (With fixed save_report)
# =============================================================================

class SignLanguageTrainer:
    """Complete trainer with fixed encoding."""
    
    def __init__(self, mode='asl'):
        self.mode = mode.lower()
        self.config = Config()
        
        if self.mode == 'asl':
            self.input_shape = self.config.ASL_INPUT_SHAPE
            self.model_creator = ModelArchitectures.create_asl_model
        else:
            self.input_shape = self.config.ISL_INPUT_SHAPE
            self.model_creator = ModelArchitectures.create_isl_model
        
        self.model = None
        self.history = None
        self.results = {}
    
    def prepare_data(self, X, y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded, self.config.NUM_CLASSES)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=self.config.TEST_SIZE,
            random_state=42, stratify=y_encoded
        )
        
        val_size = self.config.VAL_SIZE / (self.config.TRAIN_SIZE + self.config.VAL_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        y_train_labels = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train_labels), y=y_train_labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\nData Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_weight_dict
    
    def build_model(self):
        print(f"\n{'='*70}")
        print(f"BUILDING {self.mode.upper()} CNN MODEL")
        print(f"{'='*70}")
        
        self.model = self.model_creator(self.input_shape, self.config.NUM_CLASSES)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, class_weight_dict):
        print(f"\n{'='*70}")
        print(f"TRAINING {self.mode.upper()} MODEL")
        print(f"{'='*70}")
        
        checkpoint_path = f'{self.config.SAVE_DIR}/{self.mode}_checkpoint.keras'
        initial_epoch = 0
        
        if os.path.exists(checkpoint_path):
            print("\nFOUND CHECKPOINT! Resuming...")
            try:
                self.model = keras.models.load_model(checkpoint_path)
                history_path = f'{self.config.SAVE_DIR}/{self.mode}_history.json'
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        saved_history = json.load(f)
                        initial_epoch = len(saved_history.get('accuracy', []))
                        print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not load checkpoint, starting fresh...")
                self.model = None
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.config.SAVE_DIR}/best_{self.mode}_model.keras',
                monitor='val_accuracy', save_best_only=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor='val_accuracy', save_best_only=False, verbose=0
            ),
            keras.callbacks.CSVLogger(
                f'{self.config.LOG_DIR}/{self.mode}_training_log.csv',
                append=True if initial_epoch > 0 else False
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            initial_epoch=initial_epoch,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        return self.history
    
    def evaluate(self, X_test, y_test, label_encoder):
        print(f"\n{'='*70}")
        print(f"EVALUATING {self.mode.upper()} MODEL")
        print(f"{'='*70}")
        
        y_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        class_names = [chr(65 + i) for i in range(26)]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            y_true_class = (y_true == i).astype(int)
            y_pred_class = (y_pred == i).astype(int)
            metrics['per_class'][class_name] = {
                'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
                'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
                'f1': f1_score(y_true_class, y_pred_class, zero_division=0),
                'support': int(np.sum(y_true_class))
            }
        
        print(f"\nOVERALL METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_weighted']*100:.2f}%")
        print(f"  Precision: {metrics['precision_weighted']*100:.2f}%")
        print(f"  Recall:    {metrics['recall_weighted']*100:.2f}%")
        
        best = max(metrics['per_class'].items(), key=lambda x: x[1]['f1'])
        worst = min(metrics['per_class'].items(), key=lambda x: x[1]['f1'])
        print(f"\n  Best:  '{best[0]}' (F1: {best[1]['f1']*100:.2f}%)")
        print(f"  Worst: '{worst[0]}' (F1: {worst[1]['f1']*100:.2f}%)")
        
        self.results = metrics
        return metrics
    
    def plot_results(self, metrics):
        print(f"\nGenerating plots...")
        
        # Training history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        axes[0].plot(epochs, self.history.history['accuracy'], 'o-', color='#2ecc71', linewidth=2, markersize=3, label='Training')
        axes[0].plot(epochs, self.history.history['val_accuracy'], 's-', color='#3498db', linewidth=2, markersize=3, label='Validation')
        axes[0].set_title(f'{self.mode.upper()} Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.history.history['loss'], 'o-', color='#e74c3c', linewidth=2, markersize=3, label='Training')
        axes[1].plot(epochs, self.history.history['val_loss'], 's-', color='#f39c12', linewidth=2, markersize=3, label='Validation')
        axes[1].set_title(f'{self.mode.upper()} Loss', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOT_DIR}/{self.mode}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class F1
        letters = sorted(metrics['per_class'].keys())
        f1_scores = [metrics['per_class'][l]['f1'] * 100 for l in letters]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(letters, f1_scores, color='#3498db', alpha=0.8)
        ax.axhline(y=np.mean(f1_scores), color='red', linestyle='--', linewidth=1.5)
        ax.set_title(f'{self.mode.upper()} Per-Class F1 Scores', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1-Score (%)')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOT_DIR}/{self.mode}_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {self.config.PLOT_DIR}/")
    
    def save_model(self):
        model_path = f'{self.config.SAVE_DIR}/{self.mode}_classifier.keras'
        self.model.save(model_path)
        print(f"Model saved: {model_path}")
        
        h5_path = f'{self.config.SAVE_DIR}/{self.mode}_classifier.h5'
        self.model.save(h5_path)
        print(f"Backup saved: {h5_path}")
    
    def save_report(self, metrics):
        """FIXED: Save report with UTF-8 encoding."""
        report_path = f'{self.config.REPORT_DIR}/{self.mode}_training_report.txt'
        
        # FIX: Use utf-8 encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"{self.mode.upper()} SIGN LANGUAGE CNN TRAINING REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Test Accuracy: {metrics['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']*100:.2f}%\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']*100:.2f}%\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']*100:.2f}%\n")
            f.write(f"Recall (Weighted): {metrics['recall_weighted']*100:.2f}%\n\n")
            
            f.write("PER-CLASS F1 SCORES:\n")
            f.write("-"*40 + "\n")
            for letter in sorted(metrics['per_class'].keys()):
                m = metrics['per_class'][letter]
                bar_len = int(m['f1'] * 20)
                bar = '#' * bar_len + '-' * (20 - bar_len)
                f.write(f"  {letter}: [{bar}] {m['f1']*100:.1f}%\n")
        
        print(f"Report saved: {report_path}")
        
        # Save JSON
        results_path = f'{self.config.REPORT_DIR}/{self.mode}_results.json'
        json_results = {
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(metrics['accuracy']),
            'f1_weighted': float(metrics['f1_weighted']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_weighted': float(metrics['recall_weighted']),
            'per_class': {
                letter: {k: float(v) if k != 'support' else v for k, v in m.items()}
                for letter, m in metrics['per_class'].items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved: {results_path}")


# =============================================================================
# MAIN - SMART EXECUTION
# =============================================================================

def train_sign_language_model(mode, dataset_path, skip_if_complete=True):
    """Train model with smart skip capability."""
    config = Config()
    
    # Check if already complete
    model_status, report_status = TrainingStatus.print_status(mode)
    
    if skip_if_complete and model_status == 'complete':
        print(f"\n{'='*70}")
        print(f"{mode.upper()} MODEL ALREADY TRAINED - SKIPPING")
        print(f"{'='*70}")
        print(f"Model: {config.SAVE_DIR}/{mode}_classifier.keras exists")
        
        # If only report is missing, regenerate it
        if report_status != 'complete':
            print("Report missing - loading model to regenerate...")
            try:
                temp_model = keras.models.load_model(f'{config.SAVE_DIR}/{mode}_classifier.keras')
                print("Report cannot be fully regenerated without test data.")
                print("But model is ready for use!")
            except:
                pass
        
        return None, None
    
    print("\n" + "="*70)
    print(f"TRAINING {mode.upper()} SIGN LANGUAGE MODEL")
    print("="*70)
    
    # Prepare dataset
    preparator = DatasetPreparator(dataset_path, mode=mode, max_images=config.MAX_IMAGES_PER_LETTER)
    X, y = preparator.process_dataset()
    preparator.save_stats()
    
    # Initialize trainer
    trainer = SignLanguageTrainer(mode=mode)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_weights = \
        trainer.prepare_data(X, y)
    
    # Train
    trainer.train(X_train, y_train, X_val, y_val, class_weights)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test, label_encoder)
    
    # Generate plots
    trainer.plot_results(metrics)
    
    # Save
    trainer.save_model()
    trainer.save_report(metrics)
    
    print(f"\n{'='*70}")
    print(f"{mode.upper()} TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_weighted']*100:.2f}%")
    
    return trainer, metrics


def main():
    """Smart main - skips completed training."""
    print("""
    ======================================================================
    SIGN LANGUAGE CNN TRAINING - SMART MODE
    - Checks existing models
    - Skips completed training
    - Continues from where it stopped
    ======================================================================
    """)
    
    config = Config()
    
    # Check overall status
    print("="*50)
    print("CHECKING EXISTING TRAINING STATUS")
    print("="*50)
    
    asl_model_status, _ = TrainingStatus.print_status('asl')
    isl_model_status, _ = TrainingStatus.print_status('isl')
    
    print("\n" + "="*50)
    
    # ASL - Skip if complete
    if asl_model_status == 'complete':
        print("\n>>> ASL: Already complete - SKIPPING")
        asl_trainer, asl_metrics = None, None
    else:
        print("\n>>> ASL: Starting training...")
        asl_trainer, asl_metrics = train_sign_language_model('asl', config.ASL_DATASET_PATH, skip_if_complete=False)
    
    # ISL - Always run (not completed yet)
    if isl_model_status == 'complete':
        print("\n>>> ISL: Already complete - SKIPPING")
        isl_trainer, isl_metrics = None, None
    else:
        print("\n>>> ISL: Starting training...")
        isl_trainer, isl_metrics = train_sign_language_model('isl', config.ISL_DATASET_PATH, skip_if_complete=False)
    
    # Final status
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    print(f"ASL Model: {'✅ Ready' if asl_model_status == 'complete' else '❌ Not trained'}")
    print(f"ISL Model: {'✅ Ready' if isl_model_status == 'complete' else '❌ Not trained'}")
    print(f"\nModels in: {config.SAVE_DIR}/")
    print(f"Run: python app.py")
    print("="*70)


if __name__ == "__main__":
    main()
