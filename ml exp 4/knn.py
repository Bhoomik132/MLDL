# ====================================================
# K-NEAREST NEIGHBORS (KNN) CLASSIFICATION
# ====================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

# -------- LOAD DATASET --------
data = pd.read_csv("iris.csv")

print("Dataset Preview:")
print(data.head())

# -------- FEATURES & TARGET --------
X = data.drop("species", axis=1)
y = data["species"]

# Convert categorical labels to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- FEATURE SCALING --------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================================
# KNN MODEL
# ====================================================
k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Predict
y_pred = knn_model.predict(X_test)

# ====================================================
# PERFORMANCE EVALUATION
# ====================================================
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Convert confusion matrix to DataFrame
conf_df = pd.DataFrame(conf_matrix)

print("\n===== CONFUSION MATRIX =====")
print(conf_df)

print("\nAccuracy:", accuracy)

# ====================================================
# CHECK ACCURACY FOR DIFFERENT K VALUES
# ====================================================
print("\n===== ACCURACY FOR DIFFERENT K =====")

for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"K = {k} -> Accuracy = {acc:.2f}")