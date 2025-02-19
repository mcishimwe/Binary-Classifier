"""
Bus/Tram Binary Classifier
Ishimwe && Perov

Current code assumes folder structure:

- root
-- data
--- bus
--- tram
-- test_sample
--- bus
--- tram

However, it should be relatively easy to change the folders in the code.

Code will look into data folders, extract the feature and assign the labels based on folder name: bus and tram.
Then it will save extracted feature to csv file (so they can be loaded on next start).
These features will be used to train the model, or actually few different models. Then code program selects
the best performing model to use for testing. The files to test has to be located in test_samples/bus for bus
or test_samples/tram. But it can be changed to any location.

Although we are using MFCCs to perform the testing, we intentionally kept the extraction of all
the features we have been researching. This way you can set top_features to test some other options.
And also it helps to reproduce all the steps we have done for anyone reading report. Moreover, current code
is not optimised and should be treated as a prototype where you can easily perform various tests.

P.S. Default settings have been used for all feature extraction functions and ML models. It was assumed that
parameters can be adjusted if results will be below about 80%. However, even with all default parameters we have
achieved good results.

"""



import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

data_path = "data"
output_csv = "extended_features.csv"
mfcc_mean = ['mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean', 'mfcc_6_mean',
             'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean']

mfcc_std = ['mfcc_0_std', 'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std', 'mfcc_6_std',
            'mfcc_7_std', 'mfcc_8_std', 'mfcc_9_std', 'mfcc_10_std', 'mfcc_11_std', 'mfcc_12_std']

# NOT USED
selected_features = [
    'mfcc_7_mean',
    'chroma_0_mean',
    'mfcc_12_mean',
    'chroma_8_mean',
]

# feature used for training and prediction
top_features = mfcc_mean + mfcc_std

label_mapping = {"bus": 0, "tram": 1}
reverse_label_mapping = {0: "bus", 1: "tram"}

# to keep the model
best_model_in_memory = None


def extract_features(file_path, sr=22050, duration=5):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)

        # Compute raw feature matrices
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features = {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
            'spectral_contrast_std': np.std(spectral_contrast, axis=1),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms)
        }

        flattened_features = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                flattened_features.extend(value)
            else:
                flattened_features.append(value)

        return flattened_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# save features to csv
def process_audio_files():
    features_list = []
    labels_list = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file_path.endswith('.wav'):
                    print(f"Processing file: {file_path}")
                    features = extract_features(file_path)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(folder)

    columns = [f"mfcc_{i}_mean" for i in range(13)] + \
              [f"mfcc_{i}_std" for i in range(13)] + \
              [f"chroma_{i}_mean" for i in range(12)] + \
              [f"chroma_{i}_std" for i in range(12)] + \
              ["spectral_centroid_mean", "spectral_centroid_std",
               "spectral_bandwidth_mean", "spectral_bandwidth_std"] + \
              [f"spectral_contrast_{i}_mean" for i in range(7)] + \
              [f"spectral_contrast_{i}_std" for i in range(7)] + \
              ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]

    df = pd.DataFrame(features_list, columns=columns)
    df['label'] = [label_mapping[label] for label in labels_list]

    df.to_csv(output_csv, index=False)
    print(f"Features and labels saved to {output_csv}")
    return df


# check if there is already cvs file with features
def load_or_generate_dataset():
    if os.path.exists(output_csv):
        print(f"Loading dataset from {output_csv}")
        df = pd.read_csv(output_csv)
    else:
        print("Dataset not found. Extracting features from audio files.")
        df = process_audio_files()
    return df


def normalize_data(X_train, X_val):
    global scaler  # we need global to use the same scaler when we predict
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled


def split_data(df):
    X = df[top_features]  # select only features we want
    y = df['label']

    # we use stratification to take into account that there are more bus samples than trams
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in stratified_split.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    return X_train, X_val, y_train, y_val


# we try to numerically estimate the most important features
def perform_feature_analysis(df):
    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\n--- Correlation Analysis ---")
    correlations = {}
    for col in X_scaled.columns:
        try:
            corr, _ = pointbiserialr(X_scaled[col], y)
            correlations[col] = corr
        except Exception as e:
            print(f"Error calculating correlation for {col}: {e}")

    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 4 features by correlation:")
    for feature, corr in sorted_correlations[:4]:
        print(f"{feature}: {corr}")

    print("\n--- Random Forest Feature Importance ---")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    feature_importance_rf = pd.DataFrame({'Feature': X_scaled.columns, 'Importance': importances})
    feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
    print("Top 4 features by Random Forest:")
    print(feature_importance_rf.head(4))

    print("\n--- Mutual Information ---")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X_scaled.columns, 'Mutual_Info': mi_scores})
    mi_df = mi_df.sort_values(by='Mutual_Info', ascending=False)
    print("Top 4 features by Mutual Information:")
    print(mi_df.head(4))

    print("\n--- ANOVA F-test ---")
    f_scores, p_values = f_classif(X_scaled, y)
    f_test_df = pd.DataFrame({'Feature': X_scaled.columns, 'F_Score': f_scores, 'P_Value': p_values})
    f_test_df = f_test_df.sort_values(by='F_Score', ascending=False)
    print("Top 4 features by ANOVA F-test:")
    print(f_test_df.head(4))


# since data set is very small we can train different models, evaluate them and save best performing one
def train_and_evaluate_models_with_stratification(df):
    global best_model_in_memory

    X_train, X_val, y_train, y_val = split_data(df)

    X_train_scaled, X_val_scaled = normalize_data(X_train, X_val)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(128, 64, 32), learning_rate_init=0.001, random_state=42)
    }

    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        # validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        # evaluation
        acc = accuracy_score(y_val, y_pred)
        print(f"{model_name} - Cross-Validation Accuracy: {cv_scores.mean():.4f} | Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred))

        results[model_name] = {
            'model_instance': model,
            'cv_mean_accuracy': cv_scores.mean(),
            'val_accuracy': acc,
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }

    best_model = max(results.items(), key=lambda x: x[1]['val_accuracy'])
    print(f"Best Model: {best_model[0]} with Validation Accuracy: {best_model[1]['val_accuracy']:.4f}")
    best_model_in_memory = results[best_model[0]]['model_instance']


def predict_file(file_path):
    if best_model_in_memory is None:
        print("No trained model found in memory. Please train a model first.")
        return

    features = extract_features(file_path)
    if features is None:
        print("Failed to extract features. Please check the file.")
        return

    # we make here data frame to then pick only needed features,
    # sure we replace the functions to get only mfcc cofficients from audio file

    columns = [f"mfcc_{i}_mean" for i in range(13)] + \
              [f"mfcc_{i}_std" for i in range(13)] + \
              [f"chroma_{i}_mean" for i in range(12)] + \
              [f"chroma_{i}_std" for i in range(12)] + \
              ["spectral_centroid_mean", "spectral_centroid_std",
               "spectral_bandwidth_mean", "spectral_bandwidth_std"] + \
              [f"spectral_contrast_{i}_mean" for i in range(7)] + \
              [f"spectral_contrast_{i}_std" for i in range(7)] + \
              ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]

    features_df = pd.DataFrame([features], columns=columns)
    features_selected = features_df[top_features]

    # important to normalize with the same scaler!
    features_scaled = scaler.transform(features_selected)

    prediction = best_model_in_memory.predict(features_scaled)
    predicted_label = reverse_label_mapping[prediction[0]]
    print(f"Predicted label: {predicted_label}")


def predict_files_in_folder(folder_name):
    if not os.path.exists(folder_name):
        print(f"Folder '{folder_name}' does not exist.")
        return

    print(f"Processing files from {folder_name}")
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        if file_path.endswith('.wav'):
            predict_file(file_path)


# here you may choose what you want to do
def main():
    df = load_or_generate_dataset()
    # perform_feature_analysis(df)
    train_and_evaluate_models_with_stratification(df)
    predict_files_in_folder("test_sample/bus")
    predict_files_in_folder("test_sample/tram")


if __name__ == "__main__":
    main()
