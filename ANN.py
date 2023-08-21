import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math
from sklearn import metrics
from skimage.feature import hog

# Load the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist
#(x_train, y_train) is the training set
#(x_test, y_test) is the testing set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values of the images to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

#keras is a high-level API to build and train deep learning models
# Create a sequential model using Keras
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input images
  tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with ReLU activation
  tf.keras.layers.Dropout(0.25),                    # Dropout layer to prevent overfitting
  tf.keras.layers.Dense(10, activation='softmax')  # Output layer with softmax activation for multi-class classification
])

# Define the loss function (sparse categorical cross-entropy) for the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model with the loss function and optimizer
model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

# Define callbacks for reducing learning rate on plateau and early stopping
reduceLearningRate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.0001)
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# Train the model with the fit() function
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=100,
    callbacks=[reduceLearningRate, earlyStopping]
)

# Save the trained model
model.save('/content/drive/MyDrive/Machine Learning/ANN_MNIST/Model/test.h5')

# Plot the training loss and validation loss over each epoch
plt.plot(history.history['loss'], 'b', label='Train Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation loss')
plt.axis([0, 19, 0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(x_test)
actual = y_test

# Create a confusion matrix based on the predictions and actual labels
matrix = metrics.confusion_matrix(
    actual,
    np.argmax(predictions, axis=1)
)

# Create a heatmap of the confusion matrix using seaborn and matplotlib
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DataFrameMatrix = pd.DataFrame(matrix, columns=np.unique(labels), index=np.unique(labels))
DataFrameMatrix.index.name = 'Actual'
DataFrameMatrix.columns.name = 'Predict'
plt.figure(figsize=(10, 10))
sn.set(font_scale=1.5)
sn.heatmap(DataFrameMatrix, cmap='Reds', annot=True, annot_kws={"size": 10}, fmt="d")

# Print the classification report
print(metrics.classification_report(actual, np.argmax(predictions, axis=1), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Calculate and print the accuracy score
accuracy = metrics.accuracy_score(actual, np.argmax(predictions, axis=1))
rounded_accuracy = round(accuracy, 10)
print(rounded_accuracy)
