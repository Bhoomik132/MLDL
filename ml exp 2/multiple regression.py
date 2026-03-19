# ====================================================
# MULTIPLE LINEAR, RIDGE & LASSO REGRESSION
# ====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# -------- LOAD DATASET --------
data = pd.read_csv("Housing.csv")

# -------- SELECT MULTIPLE FEATURES --------
# You can change features based on dataset
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = data['price']

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================================
# MULTIPLE LINEAR REGRESSION
# ====================================================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

# ====================================================
# RIDGE REGRESSION
# ====================================================
ridge_model = Ridge(alpha=1.0)  # alpha = regularization strength
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

# ====================================================
# LASSO REGRESSION
# ====================================================
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

# ====================================================
# PERFORMANCE MATRIX
# ====================================================
performance = pd.DataFrame({
    "Model": ["Multiple Linear", "Ridge", "Lasso"],
    "MSE": [
        mean_squared_error(y_test, y_pred_linear),
        mean_squared_error(y_test, y_pred_ridge),
        mean_squared_error(y_test, y_pred_lasso)
    ],
    "R2 Score": [
        r2_score(y_test, y_pred_linear),
        r2_score(y_test, y_pred_ridge),
        r2_score(y_test, y_pred_lasso)
    ]
})

print("\n===== PERFORMANCE COMPARISON =====")
print(performance)

# ====================================================
# COEFFICIENT COMPARISON
# ====================================================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Linear": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\n===== COEFFICIENT COMPARISON =====")
print(coefficients)