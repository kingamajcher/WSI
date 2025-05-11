import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn

# loading model from file
model = keras.models.load_model("model.h5")

# loading data
images = np.load("images.npy")
labels = np.load("labels.npy")

# normalising data
images = images / 255.0

# making predictions
predictions = model.predict(images)
predicted_labels = np.argmax(predictions, axis=1)

# calculating accuracy
accuracy = np.mean(predicted_labels == labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# making confusion matrix
cm = tf.math.confusion_matrix(labels=labels, predictions=predicted_labels)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

# printing statistics for each digit
report = classification_report(labels, predicted_labels, digits=4)
print("\nStatistics for each number:")
print(report)