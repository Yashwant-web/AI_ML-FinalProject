# Ransomware Detection Project - Final Code
# Author: Yashwant Salunkhe
# Final Model Training and Threat Intelligence Integration
# Project: Improving Early Detection of Ransomware in IoT Devices
#
# Description: This script trains multiple machine learning models on the final combined dataset and integrates with VirusTotal for threat intelligence.
# The dataset used is 'CICIDS 2023 - CICIOT2023 Big Dataset' from the Canadian Institute for Cybersecurity.
# The dataset is available at: https://www.unb.ca/cic/datasets/malmem-2023.html
# It includes Random Forest, LightGBM, XGBoost, and a Voting Classifier ensemble.
# The script also includes an optional integration with the VirusTotal API for URL scanning and threat intelligence.
#
# Import necessary libraries
# Note: Ensure you have the required libraries installed. You can install them using pip if needed.
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost requests lightgbm imbalanced-learn


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Step 1: Load the final combined dataset
dataset_path = "C:/Users/yashw/OneDrive/Desktop/AI-ML/final_combined_dataset.csv"
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

print("\nDataset Loaded")
print("Shape:", df.shape)
print(df.head())

# Step 2: Split features and labels
X = df.drop(['Attack_Label'], axis=1)
y = df['Attack_Label']

# Step 2.1: Cleaning the dataset
print("\nCleaning dataset to remove infinities and NaNs")
X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
X.fillna(X.mean(), inplace=True)
print("Dataset cleaned successfully")

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Train Random Forest Classifier
print("\nTraining Random Forest Classifier")
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Train LightGBM Classifier
print("\nTraining LightGBM Classifier")
lgb_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
lgb_model.fit(X_train, y_train)

# Step 6: Train XGBoost Classifier
print("\nTraining XGBoost Classifier")
xgb_model = XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=8, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Step 7: Build Voting Classifier Ensemble
print("\nTraining Voting Classifier Ensemble")
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)
voting_model.fit(X_train, y_train)

# Step 8: Evaluate All Models
models = {
    "Random Forest": rf_model,
    "LightGBM": lgb_model,
    "XGBoost": xgb_model,
    "Voting Ensemble": voting_model
}

for name, model in models.items():
    print(f"\nEvaluation Report for {name}:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"{name} Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Step 9: End of Machine Learning Model Section

# Step 10: VirusTotal Threat Intelligence API Integration (Optional)

print("\nStarting VirusTotal Threat Intelligence Check")

# VirusTotal API Key (user must replace with their actual API key)
API_KEY = "88612485f39f59b28a71228762121299fd2c72c7c749018f6b7023469a226a69"

def scan_url(target_url):
    headers = {"x-apikey": API_KEY}
    data = {"url": target_url}
    response = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers, data=data)

    if response.status_code == 200:
        analysis_id = response.json()["data"]["id"]
        print(f"Scan submitted successfully. Analysis ID: {analysis_id}")
        return analysis_id
    else:
        print(f"Error submitting URL: {response.json()}")
        return None

def get_scan_report(analysis_id):
    headers = {"x-apikey": API_KEY}
    report_url = "https://www.virustotal.com/api/v3/analyses/" + analysis_id
    print("Waiting 15 seconds for VirusTotal scan results...")
    time.sleep(15)
    response = requests.get(report_url, headers=headers)

    if response.status_code == 200:
        stats = response.json()["data"]["attributes"]["stats"]
        print("\nVirusTotal Scan Results:")
        print(f"Malicious: {stats['malicious']}")
        print(f"Harmless: {stats['harmless']}")
        print(f"Suspicious: {stats['suspicious']}")
        print(f"Undetected: {stats['undetected']}")
    else:
        print(f"Error retrieving scan report: {response.json()}")

# Example usage
test_url = "http://testphp.vulnweb.com"
analysis_id = scan_url(test_url)
if analysis_id:
    get_scan_report(analysis_id)
