import os.path

import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import scipy


from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Flatten, Reshape, Input, Lambda, BatchNormalization, Dropout
from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input

BATCH_SIZE = 50
EPOCHS = 250
MAX_PRINT_LABEL = 10
DATA_PATH = 'data/aivazovsky/train_256x256/'
BUFFER_SIZE = len(glob.glob(os.path.join(DATA_PATH, '*/*.jpg')))
SAVE_PATH = './result/'
NUM_GENERATE = 50
NUM_REPEAT_DATA = 34
AUTOTUNE = tf.data.AUTOTUNE
alpha = 0.2
dropout_rate = 0.3


datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.15,
    brightness_range=(0.9, 1),
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255,
)

train_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=True,
    interpolation='bicubic',
    color_mode='rgb',
)


hidden_dim = 100

generator = tf.keras.Sequential([
    Dense(16 * 16 * 512, input_shape=(hidden_dim,), activation='relu'),
    BatchNormalization(),

    Reshape((16, 16, 512)),
    Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(256, (7, 7), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(128, (9, 9), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(64, (11, 11), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(64, (13, 13), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(3, (15, 15), strides=(1, 1), padding='same', activation='sigmoid'),

])

generator.summary()

discriminator = tf.keras.Sequential([
    Conv2D(3, (15, 15), strides=(1, 1), padding='same', input_shape=[256, 256, 3]),
    LeakyReLU(),
    Dropout(dropout_rate),

    Conv2D(64, (13, 13), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(dropout_rate),

    Conv2D(128, (11, 11), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(dropout_rate),

    Conv2D(256, (9, 9), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(dropout_rate),

    Conv2D(64, (7, 7), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(dropout_rate),

    Conv2D(32, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(dropout_rate),

    Flatten(),
    Dense(1),
])

discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-6)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)


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
    th = (BUFFER_SIZE * NUM_REPEAT_DATA) // (BATCH_SIZE * MAX_PRINT_LABEL)

    for epoch in range(1, epochs + 1):
        print(f'{epoch}/{EPOCHS}: ', end='')

        start = time.time()
        n = 0

        gen_loss_epoch = 0
        generated_samples = 0
        for image_batch in dataset:
            generated_samples += len(image_batch)
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_epoch += K.mean(gen_loss)
            if (n % th == 0): print('=', end='')
            if (generated_samples >= BUFFER_SIZE * NUM_REPEAT_DATA): break
            n += 1

        if epoch % 25 == 0:
            generator.save_weights(f'./training_checkpoints/generator_checkpoint_epoch_{epoch}.h5')
            discriminator.save_weights(f'./training_checkpoints/discriminator_checkpoint_epoch_{epoch}.h5')

        history += [gen_loss_epoch / n]
        print(': ' + str(np.array(history[-1]).item()))
        print('Время эпохи {} составляет {} секунд'.format(epoch, time.time() - start))
    generator.save(f'./training_checkpoints/generator_last_checkpoint.h5')
    discriminator.save(f'./training_checkpoints/discriminator_last_checkpoint.h5')
    history = [tensor.item() for tensor in np.array(history)]
    return history


def generate_and_save_images():
    for i in range(NUM_GENERATE):
        noise = tf.random.normal((hidden_dim,))
        generated_image = generator.predict(np.expand_dims(noise, axis=0))
        plt.imshow(generated_image[0])
        plt.axis('off')
        plt.savefig(SAVE_PATH + f'generate_pic_{i}.png')
        plt.close()


def save_history(history):
    with open('history.txt', 'w') as f:
        for item in history:
            f.write(str(item) + '\n')


def load_history():
    history = []
    with open('history.txt', 'r') as f:
        for line in f:
            history.append(float(line.strip()))

    return history


def test_generator():
    i = 0
    for batch in train_generator:
        for pic in batch:
            plt.imshow(pic)
            plt.axis('off')
            plt.savefig(f'./gege/' + f'generate_pic_{i}.png')
            i += 1
        if i >= 50: break


discriminator.load_weights('./training_checkpoints/discriminator_last_checkpoint.h5')
generator.load_weights('./training_checkpoints/generator_last_checkpoint.h5')

history = load_history()

history += train(train_generator, EPOCHS)
save_history(history)

plt.plot(history)
plt.grid(True)
plt.savefig(f'graph.png')

generate_and_save_images()

test_generator()
