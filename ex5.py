
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Load CIFAR-10
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 2. Select five classes
selected_classes = [0, 1, 2, 3, 4]
mask = np.isin(y_train, selected_classes)

x_train = x_train[mask]
y_train = y_train[mask]

# Keep a small subset for faster training
rng = np.random.default_rng(42)
indices = rng.choice(
    len(x_train), min(2000, len(x_train)), replace=False
)

x_train = x_train[indices]
y_train = y_train[indices]

# 3. Normalize pixels to [-1, 1]
x_train = x_train.astype("float32") / 127.5 - 1.0

# Text description is represented by its class ID
num_classes = len(selected_classes)
latent_dim = 100
embedding_dim = 32
image_size = 32
batch_size = 64

# 4. Build the Generator
def build_generator():
    noise_input = layers.Input(shape=(latent_dim,))
    text_input = layers.Input(shape=(), dtype="int32")

    # Learn an embedding for each text class
    text_embedding = layers.Embedding(
        num_classes, embedding_dim
    )(text_input)

    # Combine random noise with text information
    x = layers.Concatenate()(
        [noise_input, text_embedding]
    )

    x = layers.Dense(8 * 8 * 128, activation="relu")(x)
    x = layers.Reshape((8, 8, 128))(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(
        128, kernel_size=3, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(
        64, kernel_size=3, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    image_output = layers.Conv2D(
        3, kernel_size=3, padding="same",
        activation="tanh"
    )(x)

    return Model(
        [noise_input, text_input],
        image_output,
        name="Generator"
    )

# 5. Build the Discriminator
def build_discriminator():
    image_input = layers.Input(
        shape=(image_size, image_size, 3)
    )
    text_input = layers.Input(shape=(), dtype="int32")

    text_embedding = layers.Embedding(
        num_classes, embedding_dim
    )(text_input)

    # Convert text embedding into a spatial feature map
    text_features = layers.Dense(
        image_size * image_size
    )(text_embedding)

    text_features = layers.Reshape(
        (image_size, image_size, 1)
    )(text_features)

    # Check image and condition together
    x = layers.Concatenate(axis=-1)(
        [image_input, text_features]
    )

    x = layers.Conv2D(
        64, kernel_size=4, strides=2, padding="same"
    )(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        128, kernel_size=4, strides=2, padding="same"
    )(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    # Raw logits are used with
    # BinaryCrossentropy(from_logits=True)
    return Model(
        [image_input, text_input],
        output,
        name="Discriminator"
    )

generator = build_generator()
discriminator = build_discriminator()

generator.summary()
discriminator.summary()

# 6. Define optimizers and loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

g_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5
)

d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5
)

# 7. Define one training step
@tf.function
def train_step(real_images, real_labels):
    batch_size_actual = tf.shape(real_images)[0]

    noise = tf.random.normal(
        (batch_size_actual, latent_dim)
    )

    # Use matching class labels for real and fake examples
    with tf.GradientTape() as g_tape, \
         tf.GradientTape() as d_tape:

        fake_images = generator(
            [noise, real_labels], training=True
        )

        real_logits = discriminator(
            [real_images, real_labels], training=True
        )

        fake_logits = discriminator(
            [fake_images, real_labels], training=True
        )

        # Discriminator learns real=1, fake=0
        d_loss_real = cross_entropy(
            tf.ones_like(real_logits), real_logits
        )

        d_loss_fake = cross_entropy(
            tf.zeros_like(fake_logits), fake_logits
        )

        d_loss = d_loss_real + d_loss_fake

        # Generator wants fake images classified as real
        g_loss = cross_entropy(
            tf.ones_like(fake_logits), fake_logits
        )

    g_gradients = g_tape.gradient(
        g_loss, generator.trainable_variables
    )

    d_gradients = d_tape.gradient(
        d_loss, discriminator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(g_gradients, generator.trainable_variables)
    )

    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )

    return g_loss, d_loss

# 8. Prepare the dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train.astype("int32"))
)

dataset = dataset.shuffle(2000).batch(batch_size)

# 9. Train the GAN
epochs = 10

for epoch in range(epochs):
    g_losses = []
    d_losses = []

    for real_images, real_labels in dataset:
        g_loss, d_loss = train_step(
            real_images, real_labels
        )

        g_losses.append(float(g_loss))
        d_losses.append(float(d_loss))

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"G loss: {np.mean(g_losses):.4f} | "
        f"D loss: {np.mean(d_losses):.4f}"
    )

# 10. Generate images from text conditions
def generate_images(class_id, number=5):
    noise = tf.random.normal((number, latent_dim))

    labels = tf.fill(
        (number,), tf.cast(class_id, tf.int32)
    )

    generated = generator(
        [noise, labels], training=False
    ).numpy()

    # Convert [-1, 1] to [0, 1] for display
    generated = (generated + 1.0) / 2.0

    fig, axes = plt.subplots(
        1, number, figsize=(2 * number, 3)
    )

    for i, ax in enumerate(axes):
        ax.imshow(np.clip(generated[i], 0, 1))
        ax.axis("off")

    fig.suptitle(
        f'Text: "a photo of a {class_names[class_id]}"'
    )
    plt.tight_layout()
    plt.show()

# 11. Generate images for a selected text label
generate_images(class_id=3, number=5)  # cat
