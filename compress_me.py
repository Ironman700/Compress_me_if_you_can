import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Reshape, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, axis=-1)

# Print the shape of the training data
print(x_train.shape)

# Encoder
def build_encoder():
    inputs = Input(shape=(28, 28, 1))
    x = Flatten()(inputs)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    encoded = Dense(32, activation='relu')(x)
    return Model(inputs, encoded, name="encoder")

# Decoder
def build_decoder():
    inputs = Input(shape=(32,))
    x = Dense(64)(inputs)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(784, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)
    return Model(inputs, decoded, name="decoder")

# Autoencoder
def build_autoencoder():
    encoder = build_encoder()
    decoder = build_decoder()
    inputs = Input(shape=(28, 28, 1))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = Model(inputs, decoded, name="autoencoder")
    return autoencoder

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Generator
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

generator = build_generator()
generator.summary()

# Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
discriminator.summary()

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    img = generator(gan_input)
    gan_output = discriminator(img)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    return gan

gan = build_gan(generator, discriminator)
gan.summary()

# Training the GAN
def train_gan(generator, discriminator, gan, epochs=10000, batch_size=128):
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_imgs = x_train[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_labels = np.ones((batch_size, 1))
        
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

train_gan(generator, discriminator, gan)

# Generate and display images
def display_generated_images(generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale images 0 - 1
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

display_generated_images(generator)

# Modified Generator
def build_modified_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

modified_generator = build_modified_generator()
gan_modified = build_gan(modified_generator, discriminator)
train_gan(modified_generator, discriminator, gan_modified)
