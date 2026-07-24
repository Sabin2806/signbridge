"""
=============================================================================
ISL SIGN LANGUAGE CNN TRAINING PIPELINE
Dataset: Prathum Arikeri (1200 images per letter)
Features: MediaPipe Hand Landmarks (126 features per sample)
Model: Convolutional Neural Network
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for ISL training."""
    
    # Paths
    ISL_DATASET_PATH = 'D:/ISL MODEL/isl dataset'  # Prathum Arikeri dataset
    SAVE_DIR = './saved_models'
    LOG_DIR = './logs'
    PLOT_DIR = './plots'
    REPORT_DIR = './reports'
    
    # Dataset
    MAX_IMAGES_PER_LETTER = 1200  # Full dataset
    RANDOM_SEED = 42
    
    # Model
    NUM_CLASSES = 26
    INPUT_SHAPE = (126,)  # 2 hands × 21 landmarks × 3 coords
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 60  # More epochs for larger dataset
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 12
    REDUCE_LR_PATIENCE = 6
    
    # Data split
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_FACTOR = 3  # More augmentation for better generalization
    
    def __init__(self):
        for directory in [self.SAVE_DIR, self.LOG_DIR, self.PLOT_DIR, self.REPORT_DIR]:
            os.makedirs(directory, exist_ok=True)


# =============================================================================
# DATASET PREPARATION
# =============================================================================

class ISLDatasetPreparator:
    """Prepares ISL dataset with landmark extraction."""
    
    def __init__(self, dataset_path, max_images=1200):
        self.dataset_path = dataset_path
        self.max_images = max_images
        self.mp_hands = mp.solutions.hands
        self.max_hands = 2
        self.features_per_sample = 126
        
        self.stats = {
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
        print(f"📏 TRIMMING ISL DATASET")
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
                print(f"  {letter}: {len(image_files)} → {self.max_images} (trimmed)")
            else:
                selected_files = image_files
                self.stats['per_class'][letter]['after_trim'] = len(image_files)
                print(f"  {letter}: {len(image_files)} (using all)")
            
            letter_files[letter] = {'path': letter_path, 'files': selected_files}
        
        self.stats['total_after_trimming'] = sum(
            self.stats['per_class'][l]['after_trim'] for l in self.stats['per_class']
        )
        
        print(f"\n📊 Summary: {self.stats['total_original_images']} → {self.stats['total_after_trimming']}")
        
        return letter_files
    
    def extract_landmarks(self, image_path):
        """Extract MediaPipe landmarks for ISL (dual hand)."""
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
            
            return self._extract_dual_hand(results), True
    
    def _extract_dual_hand(self, results):
        """Extract 126 features for dual hand ISL."""
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
        """Convert 21 landmarks to 63 normalized features."""
        features = []
        wrist = hand_landmarks.landmark[0]
        
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
            features.append(lm.z - wrist.z)
        
        return np.array(features, dtype=np.float32)
    
    def augment_landmarks(self, features, num_augmentations=3):
        """Create augmented versions of landmarks."""
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
        """Process entire ISL dataset."""
        letter_files = self.trim_dataset()
        
        print(f"\n{'='*70}")
        print(f"🔍 EXTRACTING ISL LANDMARKS")
        print(f"{'='*70}")
        
        X, y = [], []
        
        for letter, data in letter_files.items():
            print(f"\n📂 Processing '{letter}' ({len(data['files'])} images)...")
            
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
        """Print dataset statistics."""
        print(f"\n{'='*70}")
        print(f"📊 ISL DATASET STATISTICS")
        print(f"{'='*70}")
        
        print(f"\nOverall:")
        print(f"  Original images:        {self.stats['total_original_images']}")
        print(f"  After trimming:         {self.stats['total_after_trimming']}")
        print(f"  Successfully extracted: {self.stats['total_after_extraction']}")
        print(f"  Failed extractions:     {self.stats['failed_extractions']}")
        
        if Config.USE_AUGMENTATION:
            print(f"  Augmentation factor:    {Config.AUGMENTATION_FACTOR}x")
            print(f"  Total with augment:     {self.stats['total_after_augmentation']}")
        
        print(f"\n📊 Per-Class Breakdown:")
        print(f"{'Letter':<8} {'Orig':<8} {'Trim':<8} {'Extr':<8} {'Fail':<8} {'Rate':<8}")
        print("-" * 48)
        
        for letter in sorted(self.stats['per_class'].keys()):
            stats = self.stats['per_class'][letter]
            rate = (stats['extracted'] / stats['after_trim'] * 100) if stats['after_trim'] > 0 else 0
            print(f"{letter:<8} {stats['original']:<8} {stats['after_trim']:<8} "
                  f"{stats['extracted']:<8} {stats['failed']:<8} {rate:.1f}%")
    
    def save_stats(self):
        """Save statistics to JSON."""
        stats_path = f'{Config.REPORT_DIR}/isl_dataset_stats.json'
        stats_copy = self.stats.copy()
        stats_copy['per_class'] = dict(stats_copy['per_class'])
        
        with open(stats_path, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        print(f"\n✅ Stats saved: {stats_path}")


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def create_isl_model(input_shape=(126,), num_classes=26):
    """ISL CNN Model: 126 features → CNN → 26 classes."""
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='landmark_input'),
        layers.Reshape((42, 3, 1), name='reshape'),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(name='bn1_2'),
        layers.MaxPooling2D((2, 1), name='pool1'),
        layers.Dropout(0.3, name='drop1'),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(name='bn2_2'),
        layers.MaxPooling2D((2, 1), name='pool2'),
        layers.Dropout(0.3, name='drop2'),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.BatchNormalization(name='bn3_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.BatchNormalization(name='bn3_2'),
        layers.MaxPooling2D((2, 1), name='pool3'),
        layers.Dropout(0.3, name='drop3'),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.BatchNormalization(name='bn4_1'),
        layers.GlobalAveragePooling2D(name='global_pool'),
        layers.Dropout(0.4, name='drop4'),
        
        # Dense
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(0.5, name='drop5'),
        
        layers.Dense(256, activation='relu', name='dense2'),
        layers.BatchNormalization(name='bn_dense2'),
        layers.Dropout(0.5, name='drop6'),
        
        layers.Dense(128, activation='relu', name='dense3'),
        layers.BatchNormalization(name='bn_dense3'),
        layers.Dropout(0.5, name='drop7'),
        
        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    return model


# =============================================================================
# TRAINER
# =============================================================================

class ISLTrainer:
    """Complete ISL trainer with metrics and checkpoint resume."""
    
    def __init__(self):
        self.config = Config()
        self.model = None
        self.history = None
        self.results = {}
    
    def prepare_data(self, X, y):
        """Split data into train/val/test."""
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded, self.config.NUM_CLASSES)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical,
            test_size=self.config.TEST_SIZE,
            random_state=42,
            stratify=y_encoded
        )
        
        val_size = self.config.VAL_SIZE / (self.config.TRAIN_SIZE + self.config.VAL_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        y_train_labels = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\n📊 Data Split:")
        print(f"   Training:   {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test:       {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_weight_dict
    
    def build_model(self):
        """Build and compile ISL CNN model."""
        print(f"\n{'='*70}")
        print(f"🏗️  BUILDING ISL CNN MODEL")
        print(f"{'='*70}")
        
        self.model = create_isl_model(
            input_shape=self.config.INPUT_SHAPE,
            num_classes=self.config.NUM_CLASSES
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        total_params = self.model.count_params()
        print(f"\n📐 Total parameters: {total_params:,}")
        print(f"📐 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, class_weight_dict):
        """Train with checkpoint resume support."""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING ISL MODEL")
        print(f"{'='*70}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print(f"{'='*70}\n")
        
        checkpoint_path = f'{self.config.SAVE_DIR}/isl_checkpoint.keras'
        initial_epoch = 0
        
        if os.path.exists(checkpoint_path):
            print("📂 Found checkpoint! Resuming...")
            try:
                self.model = keras.models.load_model(checkpoint_path)
                history_path = f'{self.config.SAVE_DIR}/isl_history.json'
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        saved_history = json.load(f)
                        initial_epoch = len(saved_history.get('accuracy', []))
                        print(f"✅ Resuming from epoch {initial_epoch}")
            except:
                print("⚠️ Could not load checkpoint, starting fresh...")
                self.model = None
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.config.SAVE_DIR}/best_isl_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=False,
                verbose=0
            ),
            keras.callbacks.CSVLogger(
                f'{self.config.LOG_DIR}/isl_training_log.csv',
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
        """Comprehensive evaluation."""
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING ISL MODEL")
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
            'confusion_matrix': confusion_matrix(y_true, y_pred),
        }
        
        # Per-class metrics
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
        
        # Print results
        print(f"\n📈 OVERALL METRICS:")
        print("-" * 50)
        print(f"Accuracy:              {metrics['accuracy']*100:.2f}%")
        print(f"Precision (Macro):     {metrics['precision_macro']*100:.2f}%")
        print(f"Precision (Weighted):  {metrics['precision_weighted']*100:.2f}%")
        print(f"Recall (Macro):        {metrics['recall_macro']*100:.2f}%")
        print(f"Recall (Weighted):     {metrics['recall_weighted']*100:.2f}%")
        print(f"F1-Score (Macro):      {metrics['f1_macro']*100:.2f}%")
        print(f"F1-Score (Weighted):   {metrics['f1_weighted']*100:.2f}%")
        
        print(f"\n📊 PER-CLASS F1 SCORES:")
        print("-" * 50)
        for letter in sorted(metrics['per_class'].keys()):
            m = metrics['per_class'][letter]
            bar_len = int(m['f1'] * 20)
            bar = '#' * bar_len + '-' * (20 - bar_len)
            print(f"  {letter}: [{bar}] {m['f1']*100:.1f}%")
        
        best = max(metrics['per_class'].items(), key=lambda x: x[1]['f1'])
        worst = min(metrics['per_class'].items(), key=lambda x: x[1]['f1'])
        print(f"\n🏆 Best:  '{best[0]}' (F1: {best[1]['f1']*100:.2f}%)")
        print(f"⚠️  Worst: '{worst[0]}' (F1: {worst[1]['f1']*100:.2f}%)")
        
        self.results = metrics
        return metrics
    
    def plot_results(self, metrics):
        """Generate visualization plots."""
        print(f"\n📊 Generating plots...")
        
        # 1. Training history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        axes[0].plot(epochs, self.history.history['accuracy'], 'o-', color='#2ecc71', 
                    linewidth=2, markersize=3, label='Training')
        axes[0].plot(epochs, self.history.history['val_accuracy'], 's-', color='#3498db', 
                    linewidth=2, markersize=3, label='Validation')
        axes[0].set_title('ISL Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.history.history['loss'], 'o-', color='#e74c3c', 
                    linewidth=2, markersize=3, label='Training')
        axes[1].plot(epochs, self.history.history['val_loss'], 's-', color='#f39c12', 
                    linewidth=2, markersize=3, label='Validation')
        axes[1].set_title('ISL Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOT_DIR}/isl_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrix
        cm = metrics['confusion_matrix']
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        class_names = [chr(65+i) for i in range(26)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('ISL Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('ISL Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOT_DIR}/isl_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-class F1
        letters = sorted(metrics['per_class'].keys())
        f1_scores = [metrics['per_class'][l]['f1'] * 100 for l in letters]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(letters, f1_scores, color='#00ff66', alpha=0.8)
        ax.axhline(y=np.mean(f1_scores), color='red', linestyle='--', linewidth=1.5)
        ax.set_title('ISL Per-Class F1 Scores', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1-Score (%)')
        ax.set_xlabel('Letter')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOT_DIR}/isl_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to: {self.config.PLOT_DIR}/")
    
    def save_model(self):
        """Save trained model."""
        model_path = f'{self.config.SAVE_DIR}/isl_classifier.keras'
        self.model.save(model_path)
        print(f"\n✅ Model saved: {model_path}")
        
        h5_path = f'{self.config.SAVE_DIR}/isl_classifier.h5'
        self.model.save(h5_path)
        print(f"✅ Backup saved: {h5_path}")
    
    def save_report(self, metrics):
        """Save training report."""
        report_path = f'{self.config.REPORT_DIR}/isl_training_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ISL SIGN LANGUAGE CNN TRAINING REPORT\n")
            f.write(f"Dataset: Prathum Arikeri (1200 images/letter)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Input Shape: {self.config.INPUT_SHAPE}\n")
            f.write(f"Output Classes: {self.config.NUM_CLASSES}\n")
            f.write(f"Max Images/Letter: {self.config.MAX_IMAGES_PER_LETTER}\n")
            f.write(f"Batch Size: {self.config.BATCH_SIZE}\n")
            f.write(f"Epochs Trained: {len(self.history.history['loss'])}\n")
            f.write(f"Augmentation: {self.config.AUGMENTATION_FACTOR}x\n\n")
            
            f.write("RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Test Accuracy: {metrics['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']*100:.2f}%\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']*100:.2f}%\n\n")
            
            f.write("PER-CLASS F1 SCORES:\n")
            f.write("-"*40 + "\n")
            for letter in sorted(metrics['per_class'].keys()):
                m = metrics['per_class'][letter]
                bar_len = int(m['f1'] * 20)
                bar = '#' * bar_len + '-' * (20 - bar_len)
                f.write(f"  {letter}: [{bar}] {m['f1']*100:.1f}%\n")
        
        print(f"✅ Report saved: {report_path}")
        
        # Save JSON
        results_path = f'{self.config.REPORT_DIR}/isl_results.json'
        json_results = {
            'mode': 'ISL',
            'dataset': 'Prathum Arikeri',
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
        
        print(f"✅ Results saved: {results_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training execution."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ISL SIGN LANGUAGE CNN TRAINING                           ║
    ║     Dataset: Prathum Arikeri (1200 images/letter)            ║
    ║     Features: MediaPipe Dual-Hand Landmarks (126 features)   ║
    ║     Model: Convolutional Neural Network                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    config = Config()
    
    print("="*70)
    print("🚀 STARTING ISL TRAINING")
    print("="*70)
    
    # Prepare dataset
    preparator = ISLDatasetPreparator(
        config.ISL_DATASET_PATH, 
        max_images=config.MAX_IMAGES_PER_LETTER
    )
    X, y = preparator.process_dataset()
    preparator.save_stats()
    
    # Initialize trainer
    trainer = ISLTrainer()
    
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
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ ISL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_weighted']*100:.2f}%")
    print(f"   Precision: {metrics['precision_weighted']*100:.2f}%")
    print(f"   Recall:    {metrics['recall_weighted']*100:.2f}%")
    print(f"\n📁 Model saved: {config.SAVE_DIR}/isl_classifier.keras")
    print(f"📁 Report saved: {config.REPORT_DIR}/isl_training_report.txt")
    print(f"\n{'='*70}")
    
    return trainer, metrics

if __name__ == "__main__":
    main()
