import os
import random
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Set random seed for consistent results
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_random_seed(42)

# Function to extract audio features
def extract_features(audio, sample_rate, mfcc=True, chroma=True, mel=True):
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

    return result

# Function to augment audio data
def augment_audio(audio, sample_rate):
    augmented_audios = []

    # Time-shifting
    shift = np.random.randint(sample_rate)
    audio_shifted = np.roll(audio, shift)
    augmented_audios.append(audio_shifted)

    # Pitch-shifting
    try:
        pitch_shift = np.random.uniform(-5, 5)
        audio_pitched = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
        augmented_audios.append(audio_pitched)
    except Exception as e:
        print(f"Error in pitch shifting: {str(e)}")

    # Adding noise
    noise = np.random.randn(len(audio))
    audio_noisy = audio + 0.005 * noise
    augmented_audios.append(audio_noisy)

    # Time-stretching
    try:
        stretch_rate = np.random.uniform(0.8, 1.2)
        audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
        if len(audio_stretched) >= sample_rate:
            augmented_audios.append(audio_stretched)
    except Exception as e:
        print(f"Error in time stretching: {str(e)}")

    # Ensure augmented audios are not too short
    min_length = sample_rate  # Minimum length is 1 second
    valid_audios = [aug_audio for aug_audio in augmented_audios if len(aug_audio) >= min_length]
    
    return valid_audios

# Function to process audio files
def process_audio_file(file_path, folder):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        features = extract_features(audio, sample_rate)
        if features is None:
            return None

        augmented_audios = augment_audio(audio, sample_rate)
        augmented_features = []

        for aug_audio in augmented_audios:
            aug_features = extract_features(aug_audio, sample_rate)
            augmented_features.append(aug_features)

        augmented_features = [features] + augmented_features

        labels = [folder] * len(augmented_features)
        return augmented_features, labels
    except Exception as e:
        print(f"Error encountered while parsing file {file_path}: {str(e)}")
        return None

# Path to dataset
data_path = "NigensAudio"

# Lists to store features and labels
X = []
y = []

# Loop through each subfolder in the dataset path
audio_files = [(os.path.join(data_path, folder, file), folder)
               for folder in os.listdir(data_path)
               if os.path.isdir(os.path.join(data_path, folder))
               for file in os.listdir(os.path.join(data_path, folder))
               if file.endswith('.wav')]

# Process audio files in parallel
results = Parallel(n_jobs=-1)(delayed(process_audio_file)(file_path, folder) for file_path, folder in audio_files)

# Collect results
for result in results:
    if result is not None:
        features, labels = result
        X.extend(features)
        y.extend(labels)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert class labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test, le

# Split data into training and testing sets
X_train, X_test, y_train, y_test, le = split_data(X, y)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Start timing
start_time = time.time()

# Train model
rf_model.fit(X_train, y_train)

# Stop timing
end_time = time.time()
training_time = end_time - start_time

# Evaluate model on training data
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate model on testing data
start_time = time.time()
y_test_pred = rf_model.predict(X_test)
end_time = time.time()
testing_time = end_time - start_time

test_accuracy = accuracy_score(y_test, y_test_pred)

# Function to calculate TP, TN, FP, FN for each class
def calculate_metrics(cm, class_index):
    TP = cm[class_index, class_index]
    FN = cm[class_index, :].sum() - TP
    FP = cm[:, class_index].sum() - TP
    TN = cm.sum() - (TP + FN + FP)
    return TP, TN, FP, FN

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Calculate and display metrics for each class
metrics = {}
for i in range(len(cm)):
    TP, TN, FP, FN = calculate_metrics(cm, i)
    metrics[le.classes_[i]] = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Accuracy': (TP + TN) / (TP + TN + FP + FN),
        'Precision': TP / (TP + FP) if (TP + FP) != 0 else 0,
        'Recall': TP / (TP + FN) if (TP + FN) != 0 else 0,
        'F1-Score': 2 * (TP / (TP + FP) * TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN)) if (TP + FP) != 0 and (TP + FN) != 0 else 0
    }

for label, metric in metrics.items():
    print(f"Class: {label}")
    for key, value in metric.items():
        print(f"  {key}: {value:.2f}")

# Calculate overall metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
micro_precision = precision_score(y_test, y_test_pred, average='micro')
micro_recall = recall_score(y_test, y_test_pred, average='micro')
micro_f1_score = f1_score(y_test, y_test_pred, average='micro')
macro_precision = precision_score(y_test, y_test_pred, average='macro')
macro_recall = recall_score(y_test, y_test_pred, average='macro')
macro_f1_score = f1_score(y_test, y_test_pred, average='macro')
weighted_precision = precision_score(y_test, y_test_pred, average='weighted')
weighted_recall = recall_score(y_test, y_test_pred, average='weighted')
weighted_f1_score = f1_score(y_test, y_test_pred, average='weighted')

# Display evaluation results
print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Testing accuracy: {test_accuracy:.2f}")
print(f"Micro Precision: {micro_precision:.2f}")
print(f"Micro Recall: {micro_recall:.2f}")
print(f"Micro F1-Score: {micro_f1_score:.2f}")
print(f"Macro Precision: {macro_precision:.2f}")
print(f"Macro Recall: {macro_recall:.2f}")
print(f"Macro F1-Score: {macro_f1_score:.2f}")
print(f"Weighted Precision: {weighted_precision:.2f}")
print(f"Weighted Recall: {weighted_recall:.2f}")
print(f"Weighted F1-Score: {weighted_f1_score:.2f}")
print(f"Time taken for training: {training_time:.2f} seconds")
print(f"Time taken for testing: {testing_time:.2f} seconds")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Display confusion matrix for training data
print("Confusion Matrix for Training Data")
plot_confusion_matrix(y_train, y_train_pred, le)

# Display confusion matrix for testing data
print("Confusion Matrix for Testing Data")
plot_confusion_matrix(y_test, y_test_pred, le)
