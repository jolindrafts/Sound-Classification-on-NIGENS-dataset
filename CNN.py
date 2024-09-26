#CNN
import os
import random
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


# Set random seed to ensure consistent results
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert class labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Convert class labels to categorical
    y_train = to_categorical(y_train, num_classes=len(np.unique(y)))
    y_test = to_categorical(y_test, num_classes=len(np.unique(y)))

    return X_train, X_test, y_train, y_test, le

# Split data into training and testing sets
X_train, X_test, y_train, y_test, le = split_data(X, y)

# Reshape data to fit CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Define input size
input_shape = (X_train.shape[1], 1, 1)

# Build CNN model with a simpler architecture
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.5))  # Increase dropout

model.add(Conv2D(64, kernel_size=(3, 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.5))  # Increase dropout

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Increase dropout
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Start timing
start_time = time.time()

# Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Calculate training time
end_time = time.time()
training_time = end_time - start_time

# Calculate testing time
start_test_time = time.time()
test_loss, test_acc = model.evaluate(X_test, y_test)
end_test_time = time.time()
testing_time = end_test_time - start_test_time

# Display accuracy as percentages
train_loss, train_acc = model.evaluate(X_train, y_train)
train_acc_percent = train_acc * 100
test_acc_percent = test_acc * 100

print(f"Training accuracy: {train_acc_percent:.2f}%")
print(f"Testing accuracy: {test_acc_percent:.2f}%")
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")

# Get predictions for test data
y_pred_train = np.argmax(model.predict(X_train), axis=1)
y_pred_test = np.argmax(model.predict(X_test), axis=1)

# Convert categorical labels back to integers
y_true_train = np.argmax(y_train, axis=1)
y_true_test = np.argmax(y_test, axis=1)

# Function to calculate TP, TN, FP, FN for each class
def calculate_metrics(cm, class_index):
    TP = cm[class_index, class_index]
    FN = cm[class_index, :].sum() - TP
    FP = cm[:, class_index].sum() - TP
    TN = cm.sum() - (TP + FN + FP)
    return TP, TN, FP, FN

# Calculate confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)

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

# Display metrics for each class
for cls, metric in metrics.items():
    print(f"Class: {cls}")
    print(f"  True Positive (TP): {metric['TP']}")
    print(f"  True Negative (TN): {metric['TN']}")
    print(f"  False Positive (FP): {metric['FP']}")
    print(f"  False Negative (FN): {metric['FN']}")
    print(f"  Accuracy: {metric['Accuracy']:.2f}")
    print(f"  Precision: {metric['Precision']:.2f}")
    print(f"  Recall: {metric['Recall']:.2f}")
    print(f"  F1-Score: {metric['F1-Score']:.2f}\n")

# Display classification report for test data
report = classification_report(y_true_test, y_pred_test, target_names=le.classes_, output_dict=True)
sensitivity = report['weighted avg']['recall']
precision = report['weighted avg']['precision']
f1_score = report['weighted avg']['f1-score']

print(f"Overall Precision: {precision:.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Display confusion matrix for training data
print("Confusion Matrix for Training Data")
plot_confusion_matrix(y_true_train, y_pred_train, le)

# Display confusion matrix for testing data
print("Confusion Matrix for Testing Data")
plot_confusion_matrix(y_true_test, y_pred_test, le)
 