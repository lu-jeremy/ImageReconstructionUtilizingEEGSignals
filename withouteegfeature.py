# std libraries
import os
import time
import tarfile
import sys
import gc

# processing libraries
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

# ml imports
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, LSTM, RepeatVector, Concatenate
from keras.layers import Dropout, Flatten, Dense, Reshape, Activation, GaussianNoise, TimeDistributed, ZeroPadding2D
from keras.layers import Conv2DTranspose, UpSampling2D, Input
from keras.models import Sequential, load_model
from keras import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model, get_file
from tensorflow.keras.optimizers import Adam


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

'''
Hyperparameters
'''
NUM_EPOCH = 100_000
INIT_LR = 0.002
BATCH_SIZE = 4
ENC_EPOCHS = 300

'''
Glob vars
'''
DATA_CAP = None
IMG_SHAPE = (256, 256, 3)
SPECT_IMG_SHAPE = (64, 64, 3)
DS = 2
PATH = r'C:\Users\bluet\Downloads\ILSVRC2013_DET_train'
DATASET_DIR = 'ILSVRC2013_DET_train'


start_time = time.time()


'''
Load all face images
'''
classes = []

AMT_FULL = 5_000
d1 = []

for root, dir, files in os.walk(PATH):
    for i, file_name in enumerate(files):

        p = os.path.join(root, file_name)

        img_ = plt.imread(p, 0)
        if img_.ndim == 3 and img_.shape[2] == 3:
            # interpolation
            img_ = cv2.resize(img_, (IMG_SHAPE[0], IMG_SHAPE[1]), interpolation=cv2.INTER_CUBIC)
            assert img_.shape == IMG_SHAPE
            d1.append(img_)

        if len(d1) > AMT_FULL:
            print('in', len(d1))
            break
    else:
        gc.collect()
        print(len(d1))
        continue
    break


end_time = time.time() - start_time
print('duration of preprocessing:', end_time)

d1 = np.array(d1)
print('shape X:', d1.shape)


'''
GAN
'''
NOISE_SIZE = 100
GEN_SHAPE = (NOISE_SIZE,)

def discriminator():
  k = [64, 128, 256, 512]
  const = max_norm(1.0)
  init = RandomNormal(stddev=0.02)

  x0 = Input(IMG_SHAPE)

  d = Conv2D(k[0], kernel_size=3, kernel_constraint=const, strides=2, padding="same")(x0)
  d = LeakyReLU(alpha=0.2)(d)
  d = Dropout(0.25)(d)
  d = BatchNormalization(momentum=0.8)(d)

  d = Conv2D(k[1], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
  d = BatchNormalization(momentum=0.8)(d)
  d = LeakyReLU(alpha=0.2)(d)

  d = Conv2D(k[2], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
  d = BatchNormalization(momentum=0.8)(d)
  d = LeakyReLU(alpha=0.2)(d)

#   d = Conv2D(k[3], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
#   d = BatchNormalization(momentum=0.8)(d)
#   d = LeakyReLU(alpha=0.2)(d)
  d = Dropout(0.25)(d)

  d = Flatten()(d)

  d = Dense(1024, activation='relu')(d)
  d = Dense(1, activation='sigmoid')(d)

  d = Model(x0, d)

  d.summary()

  return d


def generator():
  units = IMG_SHAPE[0] * 32 * 32
  k_order = [64, 128, 256]
  init = RandomNormal(stddev=0.02)
  const = max_norm(1.0)

  x0 = Input(shape=GEN_SHAPE)

  g = Dense(units, kernel_initializer=init, activation='relu')(x0)
  g = Reshape((32, 32, IMG_SHAPE[0]))(g)
  g = BatchNormalization(momentum=0.8)(g)

  for k in k_order:
    # g = UpSampling2D()(g)
    # g = Conv2D(k, kernel_size=3, kernel_initializer=init, kernel_constraint=const, padding='same')(g)
    g = Conv2DTranspose(k, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(momentum=0.8)(g)
    g = Activation('relu')(g)
    # g = Dropout(0.2)(g)
  
  g = Conv2D(16, kernel_size=3, padding='same')(g)
  g = BatchNormalization(momentum=0.8)(g)
  g = Activation('relu')(g)

  g = Conv2D(3, kernel_size=3, padding='same')(g)
  g = Activation('tanh')(g)
  g = Model(x0, g)

  g.summary()

  return g


d_opt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCH)
d = discriminator()
d.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])


g = generator()

d.trainable = False

adv_opt = Adam(lr=INIT_LR, beta_1=0.2, decay=INIT_LR / NUM_EPOCH)

gan_input = Input(shape=GEN_SHAPE)
gan_output = d(g(gan_input))
adv = Model(gan_input, gan_output)

adv.compile(loss='binary_crossentropy', optimizer=adv_opt)


figure_num = 0

for i in range(NUM_EPOCH):
    '''
    Train discriminator
    '''
    # concatenate latent features and noise
    latent_vector = np.random.normal(size=(BATCH_SIZE, NOISE_SIZE))
    generated = g.predict(latent_vector)

    # real images
    # change size
    real = d1[i: i + BATCH_SIZE]

    X = np.concatenate([real, generated])
    # real labels
    y = np.concatenate([np.full(BATCH_SIZE, .9), np.full(BATCH_SIZE, .1)])
    y += .05 * np.random.random(y.shape)

    d_loss, d_acc = d.train_on_batch(X, y)

    '''
    Fool discriminator & train GAN
    '''
    # eeg signalss
    # idx = np.random.randint(0, yhat.shape[0]-BATCH_SIZE+1)
    # print('Random index of GAN:', idx)
    # adv_features = yhat[idx: idx + BATCH_SIZE]
    fake_X = np.random.normal(size=(BATCH_SIZE, NOISE_SIZE))
    fake_y = np.zeros(BATCH_SIZE)

    adv_loss = adv.train_on_batch(fake_X, fake_y)

    # print('discrim acc: {0}'.format(d_acc))
    print('discrimLoss: {0}, advLoss: {1}'.format(d_loss, adv_loss))
    # break

    if i % 10 == 0:
        print('discriminator loss:', d_loss)
        print('adversarial loss:', adv_loss)

        fig, axes = plt.subplots(DS, DS)
        fig.set_size_inches(DS, DS)
        count = 0

        # plt.imshow(cv2.resize(real[0], (256, 256)))
        for i in range(DS):
          for j in range(DS):
              axes[i, j].imshow(cv2.resize(real[count], (64, 64)))
              axes[i, j].axis('off')

              count += 1
        
        plt.savefig('predictions/real{0}.png'.format(figure_num))


        # plt.imshow(cv2.resize(generated[0], (256, 256)))
        count = 0
        for i in range(DS):
            for j in range(DS):
                axes[i, j].imshow(cv2.resize(generated[count], (64, 64)))
                axes[i, j].axis('off')

                count += 1
                
        plt.savefig('predictions/generated{0}.png'.format(figure_num))

        figure_num += 1

            
    # plt.show()
