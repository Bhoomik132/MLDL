# -------- LOAD DATASET --------
data = pd.read_csv("Housing.csv")
print("Dataset Preview:")
print(data.head())


# ====================================================
# PART 1: LINEAR REGRESSION
# ====================================================
print("\n================ LINEAR REGRESSION =================")

# Feature and target
X_lr = data[['area']]
y_lr = data['price']

# Split data
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# Predict
y_pred_lr = lr_model.predict(X_test_lr)

# ---- PERFORMANCE MATRIX (Regression) ----
linear_performance_matrix = pd.DataFrame({
    "Metric": ["Mean Squared Error", "Mean Absolute Error", "R2 Score"],
    "Value": [
        mean_squared_error(y_test_lr, y_pred_lr),
        mean_absolute_error(y_test_lr, y_pred_lr),
        r2_score(y_test_lr, y_pred_lr)
    ]
})

print("\nLinear Regression Performance Matrix:")
print(linear_performance_matrix)


# ====================================================
# PART 2: LOGISTIC REGRESSION
# ====================================================
print("\n================ LOGISTIC REGRESSION =================")

# Create binary target using median price
median_price = data['price'].median()
data['price_category'] = (data['price'] >= median_price).astype(int)

# Feature and target
X_log = data[['area']]
y_log = data['price_category']

# Split data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

# Train model
log_model = LogisticRegression()
log_model.fit(X_train_log, y_train_log)

# Predict
y_pred_log = log_model.predict(X_test_log)

# ---- CONFUSION MATRIX ----
conf_matrix_log = confusion_matrix(y_test_log, y_pred_log)
conf_matrix_log_df = pd.DataFrame(
    conf_matrix_log,
    columns=["Predicted Low (0)", "Predicted High (1)"],
    index=["Actual Low (0)", "Actual High (1)"]
)

# ---- PERFORMANCE METRICS ----
accuracy = accuracy_score(y_test_log, y_pred_log)
precision = precision_score(y_test_log, y_pred_log)
recall = recall_score(y_test_log, y_pred_log)
f1 = f1_score(y_test_log, y_pred_log)

logistic_performance_matrix = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (High Price)", "Recall (High Price)", "F1 Score (High Price)"],
    "Value": [accuracy, precision, recall, f1]
})

print("\nLogistic Regression Confusion Matrix:")
print(conf_matrix_log_df)

print("\nLogistic Regression Performance Matrix:")
print(logistic_performance_matrix)


# ====================================================
# OPTIONAL: VISUALIZATION
# ====================================================
import matplotlib.pyplot as plt

# Linear Regression Graph
plt.figure()
plt.scatter(X_test_lr, y_test_lr, label="Actual Price")
plt.plot(X_test_lr, y_pred_lr, color="red", label="Predicted Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression: Area vs Price")
plt.legend()
plt.show()

# Logistic Regression Graph (Sigmoid Curve)
y_prob_log = log_model.predict_proba(X_test_log)[:,1]  # probability of High Price
sorted_area = np.sort(X_test_log['area'])
sorted_prob = log_model.predict_proba(sorted_area.reshape(-1,1))[:,1]

plt.figure()
plt.scatter(X_test_log, y_test_log, label="Actual Class")
plt.plot(sorted_area, sorted_prob, color="red", label="Predicted Probability")
plt.xlabel("Area")
plt.ylabel("Probability (High Price)")
plt.title("Logistic Regression (Sigmoid Curve)")
plt.legend()
plt.show()
