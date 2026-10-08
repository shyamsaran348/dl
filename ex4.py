
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    RepeatVector, Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load CIFAR-10
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

y_train = y_train.flatten()
y_test = y_test.flatten()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 2. Use a small subset for a quick demonstration
rng = np.random.default_rng(42)

train_idx = rng.choice(
    len(x_train), 1000, replace=False
)
test_idx = rng.choice(
    len(x_test), 100, replace=False
)

x_train, y_train = x_train[train_idx], y_train[train_idx]
x_test, y_test = x_test[test_idx], y_test[test_idx]

# 3. Create class-based captions
def make_caption(label):
    return "startseq a photo of a " + class_names[label] + " endseq"

train_captions = [make_caption(y) for y in y_train]
test_captions = [make_caption(y) for y in y_test]

# 4. Tokenize captions
tokenizer = Tokenizer(
    filters="",
    lower=True,
    oov_token="<unk>"
)

tokenizer.fit_on_texts(train_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(
    len(c.split()) for c in train_captions
)

print("Vocabulary size:", vocab_size)
print("Maximum caption length:", max_length)

# 5. Convert captions to integer sequences
sequences = tokenizer.texts_to_sequences(train_captions)

# 6. Create teacher-forcing input/output pairs
decoder_inputs = []
next_words = []
image_indices = []

for image_idx, seq in enumerate(sequences):
    for t in range(1, len(seq)):
        decoder_inputs.append(seq[:t])
        next_words.append(seq[t])
        image_indices.append(image_idx)

decoder_inputs = pad_sequences(
    decoder_inputs,
    maxlen=max_length - 1,
    padding="post"
)

next_words = np.array(next_words)
image_indices = np.array(image_indices)

# 7. Prepare image input for the CNN
def prepare_images(images):
    images = tf.image.resize(images, (224, 224)).numpy()
    return preprocess_input(images)

x_train_prepared = prepare_images(x_train)
x_test_prepared = prepare_images(x_test)

# 8. Use pretrained ResNet50 as image encoder
cnn = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

cnn.trainable = False

train_features = cnn.predict(
    x_train_prepared, batch_size=16, verbose=1
)

test_features = cnn.predict(
    x_test_prepared, batch_size=16, verbose=1
)

print("Image feature shape:", train_features.shape)

# 9. Define CNN-LSTM decoder
image_input = Input(shape=(2048,), name="image_features")
text_input = Input(
    shape=(max_length - 1,), name="partial_caption"
)

image_projection = Dense(256, activation="relu")(image_input)
image_projection = RepeatVector(max_length - 1)(image_projection)

word_embedding = Embedding(
    input_dim=vocab_size,
    output_dim=128,
    mask_zero=True
)(text_input)

combined = Concatenate(axis=-1)(
    [image_projection, word_embedding]
)

decoder = LSTM(256)(combined)
decoder = Dropout(0.3)(decoder)
output = Dense(vocab_size, activation="softmax")(decoder)

caption_model = Model(
    inputs=[image_input, text_input],
    outputs=output
)

caption_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 10. Train the decoder
caption_model.fit(
    [
        train_features[image_indices],
        decoder_inputs
    ],
    next_words,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 11. Generate a caption one word at a time
def generate_caption(feature):
    caption = ["startseq"]

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([" ".join(caption)])
        seq = pad_sequences(
            seq, maxlen=max_length - 1, padding="post"
        )

        probs = caption_model.predict(
            [feature.reshape(1, -1), seq],
            verbose=0
        )[0]

        predicted_id = int(np.argmax(probs))

        if predicted_id == 0:
            break

        predicted_word = tokenizer.index_word.get(
            predicted_id, ""
        )

        if predicted_word == "endseq":
            break

        caption.append(predicted_word)

    return " ".join(caption[1:])

# 12. Test on an unseen image
test_number = 0

predicted_caption = generate_caption(
    test_features[test_number]
)

print("Actual class:", class_names[y_test[test_number]])
print("Expected caption:", test_captions[test_number])
print("Generated caption:", predicted_caption)

# 13. Visualize the test image
plt.figure(figsize=(4, 4))
plt.imshow(x_test[test_number])
plt.title("Input Image")
plt.axis("off")
plt.show()
