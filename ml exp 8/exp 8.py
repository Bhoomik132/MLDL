# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Step 2: Load Dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Step 3: Normalize Data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 4: Reshape (for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Step 5: Build CNN Model
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Step 6: Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train Model
model.fit(X_train, y_train, epochs=5)

# Step 8: Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Step 9: Prediction
predictions = model.predict(X_test)
print(predictions[0])
# Display first 10 images with predictions
for i in range(10):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    
    # Get predicted label
    predicted_label = np.argmax(predictions[i])
    
    # Actual label
    actual_label = y_test[i]
    
    plt.title(f"Predicted: {predicted_label}, Actual: {actual_label}")
    plt.axis('off')
    plt.show()