
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# Load pretrained VGG16
model = VGG16(weights="imagenet", include_top=False)

# Load and preprocess image
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
path = tf.keras.utils.get_file("image.jpg", url)

img = load_img(path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Select 3 convolutional layers
layers = ["block1_conv1", "block3_conv1", "block5_conv1"]

# Create model to extract feature maps
outputs = [model.get_layer(name).output for name in layers]
feature_model = tf.keras.Model(model.input, outputs)

# Get feature maps
features = feature_model.predict(x, verbose=0)

# Display feature maps
for name, fmap in zip(layers, features):
    print(name, fmap.shape)

    plt.figure(figsize=(12, 4))
    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.imshow(fmap[0, :, :, i], cmap="gray")
        plt.axis("off")

    plt.suptitle(name)
    plt.show()

# Visualize filters from first convolutional layer
filters, biases = model.get_layer("block1_conv1").get_weights()

plt.figure(figsize=(8, 4))
for i in range(8):
    f = filters[:, :, :, i]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)

    plt.subplot(2, 4, i + 1)
    plt.imshow(f)
    plt.axis("off")

plt.suptitle("Learned Convolutional Filters")
plt.show()
