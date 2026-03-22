# ====================================================
# SUPPORT VECTOR MACHINE (SVM) - dat.csv DATASET
# ====================================================

# -------- IMPORT LIBRARIES --------
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -------- LOAD DATASET --------
data = pd.read_csv("data.csv")

print("Dataset Preview:\n")
print(data.head())

# -------- FEATURES & TARGET --------
# Assuming last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert target to numeric (if needed)
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

# -------- SVM MODEL --------
svm_model = SVC()

# -------- HYPERPARAMETER TUNING --------
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(svm_model, param_grid, cv=5)
grid.fit(X_train, y_train)

# -------- BEST MODEL --------
best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# -------- PREDICTION --------
y_pred = best_model.predict(X_test)

# -------- CONFUSION MATRIX --------
conf_matrix = confusion_matrix(y_test, y_pred)

conf_df = pd.DataFrame(conf_matrix)

print("\n===== CONFUSION MATRIX =====")
print(conf_df)

# -------- PERFORMANCE --------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))