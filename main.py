import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib
import os

from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Flatten, Reshape, Input, Lambda, BatchNormalization, Dropout
from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose
from keras.utils import image_dataset_from_directory

BATCH_SIZE = 100
EPOCHS = 100
MAX_PRINT_LABEL = 10
BUFFER_SIZE = 3989
SAVE_PATH = './result/'
NUM_GENERATE = 10


x_train = image_dataset_from_directory(
    'data/train_256x256/',
    labels=None,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(256, 256),
    shuffle=True,
    interpolation='bicubic',
)

AUTOTUNE = tf.data.AUTOTUNE

normalization_layer = tf.keras.layers.Rescaling(1./255)
x_train = x_train.map(lambda x: (normalization_layer(x)))
x_train = x_train.prefetch(buffer_size=AUTOTUNE)

hidden_dim = 8 * 8 * 256


# формирование сетей

generator = tf.keras.Sequential([
    Dense(8 * 8 * 256, activation='relu', input_shape=(hidden_dim,)),
    BatchNormalization(),
    Reshape((8, 8, 256)),

    Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid'),
])

generator.summary()

# дискриминатор

discriminator = tf.keras.Sequential([
    Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=[256, 256, 3]),
    LeakyReLU(),
    Dropout(0.3),

    Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),

    Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),

    Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),

    Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),

    Flatten(),
    Dense(1)

])

discriminator.summary()

# потери
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)


# обучение
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, hidden_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    history = []

    th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

    for epoch in range(1, epochs + 1):
        print(f'{epoch}/{EPOCHS}: ', end='')

        start = time.time()
        n = 0

        gen_loss_epoch = 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_epoch += K.mean(gen_loss)
            if (n % th == 0): print('=', end='')
            n += 1

        history += [gen_loss_epoch / n]
        print(': ' + str(history[-1]))
        print('Время эпохи {} составляет {} секунд'.format(epoch, time.time() - start))

    return history


# запуск процесса обучения

history = train(x_train, EPOCHS)


def generate_and_save_images():
    for i in range(NUM_GENERATE):
        noise = tf.random.normal((hidden_dim,))
        generated_image = generator.predict(np.expand_dims(noise, axis=0))
        plt.imshow(generated_image[0])
        plt.axis('off')
        plt.savefig(SAVE_PATH + f'generate_pic_{i}.png')
        plt.close()



plt.plot(history)
plt.grid(True)
plt.savefig(r'.\graph')

generate_and_save_images()



