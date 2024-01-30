import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2


tf.config.run_functions_eagerly(True)

# Hyperparameters
image_size = (266, 218, 182)
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2
batch_size = 32  # Update with a batch size suitable for your data
ema = 0.999
learning_rate = 1e-4  
weight_decay = 1e-4
num_epochs = 10
kid_image_size = 299
max_signal_rate = 0.1
min_signal_rate = 0.01
target_height = 218  # Set your desired height
target_width = 182
# Training parameters
batch_size = 32
epochs = 10  # You can adjust this value as needed
# Set your desired width


def preprocess_image(image):
    image = tf.image.resize(image, size=(target_height, target_width), antialias=True)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return image


def prepare_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Shuffle, batch, and prefetch the dataset
    dataset = (
        dataset.shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset


def polynomial_kernel(features_1, features_2):
    feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype="float32")
    return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0


# KID Metric
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size[0], image_size[1], 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        kernel_real = polynomial_kernel(real_features, real_features)
        kernel_generated = polynomial_kernel(generated_features, generated_features)
        kernel_cross = polynomial_kernel(real_features, generated_features)

        batch_size = real_features.shape[0]
        batch_size_f = tf.cast(batch_size, dtype="float32")
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


# Diffusion Model
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.cast(tf.math.acos(max_signal_rate), "float32")
        end_angle = tf.cast(tf.math.acos(min_signal_rate), "float32")
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = tf.math.cos(diffusion_angles)
        noise_rates = tf.math.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps):
        initial_noise = keras.random.normal(
            shape=(num_images, image_size[0], image_size[1], 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, data):
        images = data[0]  
        labels = data[1]  

        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size,) + image_size)

        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )

        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates[..., tf.newaxis] * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = keras.losses.mean_absolute_error(noises, pred_noises)
            image_loss = keras.losses.mean_absolute_error(images, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}


def get_network(image_size, widths, block_depth):
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    x = inputs

    for width in widths:
        for _ in range(block_depth):
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="alignment_detection_model")

diffusion_model = DiffusionModel(
    image_size=image_size, widths=widths, block_depth=block_depth
)

diffusion_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))


early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_kid",
    patience=3,
    restore_best_weights=True,
)



def load_data(data_path, target_height, target_width):
    image_list = []
    for filename in os.listdir(data_path):
        if filename.endswith(".npy"):
            filepath = os.path.join(data_path, filename)
            image = np.load(filepath)
            image = preprocess_image(image)
            print(f"Shape of {filename}: {image.shape}")
            image_list.append(image)
    data = np.stack(image_list)
    return data


def load_labels(labels_directory, target_height, target_width):
    label_files = os.listdir(labels_directory)
    labels_list = []

    for file in label_files:
        if file.endswith(".npy"):
            filepath = os.path.join(labels_directory, file)
            label = np.load(filepath)

            crop_height = min(label.shape[0], target_height)
            crop_width = min(label.shape[1], target_width)
            cropped_label = label[:crop_height, :crop_width, :]

            if (
                cropped_label.shape[0] < target_height
                or cropped_label.shape[1] < target_width
            ):
                pad_height = max(0, target_height - cropped_label.shape[0])
                pad_width = max(0, target_width - cropped_label.shape[1])
                cropped_label = np.pad(
                    cropped_label,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode="constant",
                )


            print(f"Shape of {file}: {cropped_label.shape}")

            labels_list.append(cropped_label)


    labels_array = np.stack(labels_list)

    return labels_array


# Load your dataset 
X_train = load_data(
    "C:\\Users\\bnvsa\\OneDrive\\Desktop\\data", target_height, target_width
)
Y_train = load_labels(
    "C:\\Users\\bnvsa\\OneDrive\\Desktop\\labels", target_height, target_width
)


split = int(0.9 * len(X_train))
X_train, X_val = X_train[:split], X_train[split:]
Y_train, Y_val = Y_train[:split], Y_train[split:]

train_dataset = prepare_dataset(X_train, Y_train)
val_dataset = prepare_dataset(X_val, Y_val)
# Compile the diffusion model
diffusion_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

# Print some information about your datasets
print("Number of training samples:", len(X_train))
print("Number of validation samples:", len(X_val))
print("Shape of a training image:", X_train[0].shape)
print("Shape of a validation image:", X_val[0].shape)
print("Shape of a training label:", Y_train[0].shape)
print("Shape of a validation label:", Y_val[0].shape)


# Create a callback for early stopping
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_kid",
    patience=3,
    restore_best_weights=True,
)

# Train the diffusion model
diffusion_model.fit(
    X_train[:batch_size],
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val[:batch_size], Y_val[:batch_size]),
    callbacks=[early_stopping_callback],
)
