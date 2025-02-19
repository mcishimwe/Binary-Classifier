import os
import librosa
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(file_name):
    audio, sample_rate = librosa.load(file_path,sr=None,duration=5)
    audio = audio/np.max(np.abs(audio))

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate,n_mfcc=60)

    return np.mean(mfcc,axis=1)

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for filename in os.listdir("wav_files_training/bus"):
    file_path = os.path.join("wav_files_training/bus",filename)
    features = extract_features(file_path)
    x_train.append(features)
    y_train.append(0)

for filename in os.listdir("wav_files_training/tram"):
    file_path = os.path.join("wav_files_training/tram",filename)
    features = extract_features(file_path)
    x_train.append(features)
    y_train.append(1)

for filename in os.listdir("wav_files_validation/bus"):
    file_path = os.path.join("wav_files_validation/bus",filename)
    features = extract_features(file_path)
    x_val.append(features)
    y_val.append(0)

for filename in os.listdir("wav_files_validation/tram"):
    file_path = os.path.join("wav_files_validation/tram",filename)
    features = extract_features(file_path)
    x_val.append(features)
    y_val.append(1)

for filename in os.listdir("wav_files_testing/bus"):
    file_path = os.path.join("wav_files_testing/bus",filename)
    features = extract_features(file_path)
    x_test.append(features)
    y_test.append(0)

for filename in os.listdir("wav_files_testing/tram"):
    file_path = os.path.join("wav_files_testing/tram",filename)
    features = extract_features(file_path)
    x_test.append(features)
    y_test.append(1)

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)



y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

model = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', random_state=42))

model.fit(x_train, y_train)
y_val_pred = model.predict(x_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

y_test_pred = model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_acc * 100:.2f}%")