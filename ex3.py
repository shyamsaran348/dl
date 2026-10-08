
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

y_train = y_train.flatten()
y_test = y_test.flatten()

# 2. Select 5 classes
classes = [0, 1, 2, 3, 4]

train_idx = np.isin(y_train, classes)
test_idx = np.isin(y_test, classes)

x_train, y_train = x_train[train_idx][:100], y_train[train_idx][:100]
x_test, y_test = x_test[test_idx][:50], y_test[test_idx][:50]

# 3. Resize images
x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test = tf.image.resize(x_test, (224, 224)).numpy()

# 4. Preprocess images
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# 5. Load pretrained CNN for feature extraction
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

base_model.trainable = False

# 6. Extract features
train_features = base_model.predict(x_train, verbose=0)
test_features = base_model.predict(x_test, verbose=0)

print("Feature shape:", train_features.shape)

# 7. Train a new classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, y_train)

# 8. Predict test images
y_pred = classifier.predict(test_features)

# 9. Calculate evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(
    y_test, y_pred, average="macro", zero_division=0))
print("Recall:", recall_score(
    y_test, y_pred, average="macro", zero_division=0))
print("F1 Score:", f1_score(
    y_test, y_pred, average="macro", zero_division=0))

# 10. Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Airplane", "Car", "Bird", "Cat", "Deer"]
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
