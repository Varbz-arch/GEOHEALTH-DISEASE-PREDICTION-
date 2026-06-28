import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
import random
warnings.filterwarnings('ignore')
print("="*60)
print("GEOHEALTH - TRAINING MODEL")
print("="*60)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("dataset/Indian_Climate_Dataset_2024_2025.csv")

print("\nDataset loaded:", df.shape)

# -----------------------------
# 2. BASIC CLEANING
# -----------------------------
df = df.drop(columns=["Date", "City"], errors="ignore")
print("\n" + "="*60)
print("DATA PREPROCESSING & EXPLORATION")
print("="*60)

# 1. First 5 rows
print("\n First 5 rows:")
print(df.head())

# 2. Dataset shape
print("\n Dataset shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 3. Column names
print("\n Columns:")
print(list(df.columns))

# 4. Missing values
print("\n Missing values:")
print(df.isnull().sum())


# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------

df["Temp_Range"] = df["Temperature_Max (°C)"] - df["Temperature_Min (°C)"]
df["Humidity_Rainfall"] = df["Humidity (%)"] * df["Rainfall (mm)"]
df["Heat_Index"] = df["Temperature_Avg (°C)"] * (df["Humidity (%)"] / 100)

df["High_AQI"] = (df["AQI"] > 150).astype(int)
df["High_Rainfall"] = (df["Rainfall (mm)"] > 100).astype(int)
df["High_Humidity"] = (df["Humidity (%)"] > 75).astype(int)

# -----------------------------
# 4. ENCODING
# -----------------------------
state_encoder = LabelEncoder()
aqi_encoder = LabelEncoder()

df["State_encoded"] = state_encoder.fit_transform(df["State"])
df["AQI_Category_encoded"] = aqi_encoder.fit_transform(df["AQI_Category"])

# -----------------------------
# 5. ADD DISEASE LABEL (WITH CONTROLLED NOISE)
# -----------------------------
def assign_disease(row):

    score_heat = 0
    score_dengue = 0
    score_malaria = 0
    score_resp = 0

    temp = row["Temperature_Avg (°C)"]
    humidity = row["Humidity (%)"]
    rainfall = row["Rainfall (mm)"]
    aqi = row["AQI"]
    cloud = row["Cloud_Cover (%)"]

    # Heatstroke
    if temp > 38:
        score_heat += 0.6
    if humidity > 70:
        score_heat += 0.2

    # Dengue
    if rainfall > 120:
        score_dengue += 0.5
    if humidity > 75:
        score_dengue += 0.3

    # Malaria
    if rainfall > 80:
        score_malaria += 0.4
    if cloud > 60:
        score_malaria += 0.3

    # Respiratory
    # Respiratory (balanced logic)
    if aqi > 150:
        score_resp += 0.2
    if aqi > 200:
        score_resp += 0.3
    if humidity < 40:
        score_resp += 0.2
    if rainfall > 100:
        score_resp += 0.1

    scores = {
        "Heatstroke": score_heat,
        "Dengue": score_dengue,
        "Malaria": score_malaria,
        "Respiratory": score_resp
    }

    label = max(scores, key=scores.get)

    # CONTROLLED NOISE 
    if random.random() < 0.12:   # 12% noise
        label = random.choice(list(scores.keys()))

    return label


# APPLY LABEL
df["Disease"] = df.apply(assign_disease, axis=1)
# -----------------------------
# 6. BALANCING 
# -----------------------------
print("\nClass distribution BEFORE balancing:")
print(df["Disease"].value_counts())

min_class = df["Disease"].value_counts().min()

df = df.groupby("Disease").sample(n=min_class, random_state=42).reset_index(drop=True)

print("\nClass distribution AFTER balancing:")
print(df["Disease"].value_counts())

# ENCODING TARGET
disease_encoder = LabelEncoder()
df["Disease_encoded"] = disease_encoder.fit_transform(df["Disease"])
# 7. FEATURES
# -----------------------------
features = [
    "Temperature_Max (°C)",
    "Temperature_Min (°C)",
    "Temperature_Avg (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Wind_Speed (km/h)",
    "Pressure (hPa)",
    "Cloud_Cover (%)",
    "AQI",
    "State_encoded",
    "AQI_Category_encoded",
    "Temp_Range",
    "Humidity_Rainfall",
    "Heat_Index",
    "High_AQI",
    "High_Rainfall",
    "High_Humidity"
]

X = df[features]
y = df["Disease_encoded"]

# -----------------------------
# 8. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 9. SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 10. MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced_subsample",
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 11. EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=disease_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=disease_encoder.classes_,
            yticklabels=disease_encoder.classes_)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(8,5))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
# -----------------------------
# 12. CROSS VALIDATION
# -----------------------------
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.2%}")

# -----------------------------
# 13. SAVE MODEL
# -----------------------------
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(state_encoder, "model/state_encoder.pkl")
joblib.dump(aqi_encoder, "model/aqi_encoder.pkl")
joblib.dump(disease_encoder, "model/disease_encoder.pkl")
joblib.dump(features, "model/features.pkl")

print("\n Model saved successfully!")
print("="*60)
