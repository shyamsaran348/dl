import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications.resnet50 import (preprocess_input, decode_predictions)

from tensorflow.keras.utils import load_img,img_to_array

model=ResNet50(weights='imagenet')

image_url = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/example_images/grace_hopper.jpg"
)

image_path = tf.keras.utils.get_file(
    "sample_image.jpg",
    image_url
)
img=load_img(image_path,target_size=(224,224))
img_array=img_to_array(img)
print("Image shape before batching:", img_array.shape)
img_batch=np.expand_dims(img_array,axis=0)

print("Image shape after batching:", img_batch.shape)

preprocessed_img=preprocess_input(img_batch)

predictions=model.predict(preprocessed_img,verbose=0)

print("Prediction array shape:", predictions.shape)

top_5=decode_predictions(predictions,top=5)[0]

plt.figure(figsize=(10,5))
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()
print("\nTop-5 ImageNet predictions:")
for rank, (class_id, class_name, score) in enumerate(
    top_5, start=1
):
    print(
        f"{rank}. {class_name} "
        f"({class_id}) : {score * 100:.2f}%"
    )
