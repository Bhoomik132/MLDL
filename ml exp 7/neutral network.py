# Step 1: Import Libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
dataset = pd.read_csv("diabetics.csv", header=None)

# Step 3: Split into input and output
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 6: Build ANN Model
model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(units=8, activation='relu', input_dim=X.shape[1]))

# Hidden Layer 2
model.add(Dense(units=8, activation='relu'))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

# Step 7: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Train Model
model.fit(X_train, y_train, epochs=100, batch_size=10)

# Step 9: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Step 10: Prediction
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

print(y_pred[:10])