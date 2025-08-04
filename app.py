import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the MODIS datasets
df1 = pd.read_csv("modis_2021_India.csv")
df2 = pd.read_csv("modis_2022_India.csv")
df3 = pd.read_csv("modis_2023_India.csv")

# Combine all 3 years
df = pd.concat([df1, df2, df3], ignore_index=True)

# Print class distribution before SMOTE
print("Class distribution before SMOTE:")
print(df['type'].value_counts())

# Feature Selection
X = df[['brightness', 'scan', 'track', 'acq_time', 'latitude', 'longitude']]
y = df['type']  # target variable

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class balancing
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
