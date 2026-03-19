# ====================================================
# DECISION TREE & RANDOM FOREST CLASSIFICATION
# ====================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

# -------- LOAD DATASET --------
data = pd.read_csv("heart.csv")

print("Dataset Preview:")
print(data.head())

# -------- FEATURES & TARGET --------
X = data.drop("target", axis=1)  # independent variables
y = data["target"]               # dependent variable (0/1)

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================================
# DECISION TREE CLASSIFIER
# ====================================================
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# Confusion Matrix
conf_dt = confusion_matrix(y_test, y_pred_dt)

# Accuracy
acc_dt = accuracy_score(y_test, y_pred_dt)

# ====================================================
# RANDOM FOREST CLASSIFIER
# ====================================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Confusion Matrix
conf_rf = confusion_matrix(y_test, y_pred_rf)

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)

# ====================================================
# PRINT RESULTS
# ====================================================
import pandas as pd

# Decision Tree Confusion Matrix
conf_dt_df = pd.DataFrame(
    conf_dt,
    columns=["Predicted No Disease", "Predicted Disease"],
    index=["Actual No Disease", "Actual Disease"]
)

# Random Forest Confusion Matrix
conf_rf_df = pd.DataFrame(
    conf_rf,
    columns=["Predicted No Disease", "Predicted Disease"],
    index=["Actual No Disease", "Actual Disease"]
)

print("\n===== DECISION TREE CONFUSION MATRIX =====")
print(conf_dt_df)
print("Accuracy:", acc_dt)

print("\n===== RANDOM FOREST CONFUSION MATRIX =====")
print(conf_rf_df)
print("Accuracy:", acc_rf)