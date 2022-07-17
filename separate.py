'''
init lr
batch size
make separate file without eeg extraction
image augmentation

random/not random adversarial images
change image size

inception score

vgg
'''
# std libraries
import os
import time
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
from keras.layers import Dropout, Flatten, Dense, Reshape, Activation, GaussianNoise, TimeDistributed
from keras.layers import Conv2DTranspose, UpSampling2D, Input
from keras.models import Sequential, load_model
from keras import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model, get_file
from tensorflow.keras.optimizers import Adam
from mtcnn import MTCNN


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

'''
Hyperparameters
'''
NUM_EPOCH = 100_000
INIT_LR = 0.002
BATCH_SIZE = 2
ENC_EPOCHS = 300

'''
Glob vars
'''
DATA_CAP = None
IMG_SHAPE = (256, 256, 3)
SPECT_IMG_SHAPE = (64, 64, 3)
DS = 1
PATH = r'C:\Users\bluet\Downloads\ILSVRC2013_DET_train'
SPECT_DIR = r'C:\Users\bluet\Downloads\SRA\MindBigData-Imagenet-IN-v1.0\MindBigData-Imagenet-v1.0-Imgs'
DATASET_DIR = 'ILSVRC2013_DET_train'


start_time = time.time()

'''
EEG Data Load + Feature Extraction
- Spectrogram + CNN
'''
# always use spect as a list first then array or else it will give the wrong shape
spect = []
categories = []
cap = 0


'''
Load all face images
'''
classes = []

AMT_FULL = 40
X = []

m = MTCNN()

for root, dir, files in os.walk(PATH):
    for i, file_name in enumerate(files):
        p = os.path.join(root, file_name)

        img_ = plt.imread(p, 0)
        if img_.ndim == 3 and img_.shape[2] == 3:
            faces = m.detect_faces(img_)

            if len(faces) != 0:
                # interpolation
                img_ = cv2.resize(img_, (IMG_SHAPE[0], IMG_SHAPE[1]), interpolation=cv2.INTER_CUBIC)
                # plt.imshow(img_)
                # plt.show()

                assert img_.shape == IMG_SHAPE

                # extract img num + category
                index = file_name.find('_')
                dot_idx = file_name.find('.')
                cat = file_name[:index]
                img_num = file_name[index+1:dot_idx]

                spect_fname = 'MindBigData_Imagenet_Insight_' + cat + '_' + img_num
                # print(spect_fname)

                for root_, _, files_ in os.walk(SPECT_DIR):
                    for i_, file_name_ in enumerate(files_):
                        if spect_fname in file_name_:
                            print('in', len(X))
                            p_ = os.path.join(root_, file_name_)
                            spect_img = plt.imread(p_)
                            spect_img = np.delete(spect_img, -1, axis=2)
                            spect_img = cv2.resize(spect_img, (SPECT_IMG_SHAPE[0], SPECT_IMG_SHAPE[1]))

                            spect.append(spect_img)
                            categories.append(cat)                
                            X.append(img_)
            
            # sys.exit()
    print(len(X))
    gc.collect()
    if len(X) > AMT_FULL:
        break


end_time = time.time() - start_time
print('duration of preprocessing:', end_time)

spect = np.array(spect)
print('Spect shape', spect.shape)

X = np.array(X)
y = np.ones(X.shape[0])
print('Amt of images', X.shape)

print('len of categories', len(categories))
y_train = np.array(categories)
unique, inverse = np.unique(y_train, return_inverse=True)
y_train = np.eye(unique.shape[0])[inverse]
print('Shape of y_train', y_train.shape)
amt_classes = y_train.shape[1]




'''
ENCODER
'''
c = Sequential([
                Conv2D(32, kernel_size=3, activation='relu', strides=2, input_shape=SPECT_IMG_SHAPE),
                MaxPooling2D(pool_size=(2, 2), strides=2),

                Conv2D(64, kernel_size=3, strides=2, activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),

                # Conv2D(64, kernel_size=3, activation='relu'),
                # MaxPooling2D(pool_size=(2, 2)),

                # extra blocks to downsample size of latent vector
                # Conv2D(64, kernel_size=3, activation='relu'),
                # MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),

                Dense(128, activation='relu'),
                # Dense(64, activation='relu'),
                # Dropout(0.25),
                Dense(amt_classes, activation='sigmoid')
])

c.summary()

# sys.exit()

c.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# plot_model(c, show_shapes=True, to_file='c.png')

# separate fit from architecture description because verbose training is long
history = c.fit(spect, y_train, epochs=ENC_EPOCHS)

# plot history
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

c_enc = Model(inputs=c.inputs, outputs=c.get_layer('flatten_2').output)

c_enc.summary()

# c_enc = load_model('c_enc.h5')

# plot_model(c_enc, show_shapes=True, to_file='c_enc.png')

yhat = c_enc.predict(spect)
c_enc.save('c_enc_faces.h5')
print(yhat.shape)
print(yhat)


'''
GAN
'''
NOISE_SIZE = 100
LATENT_SIZE = yhat.shape[1]
GEN_SHAPE = (LATENT_SIZE + NOISE_SIZE,)

def discriminator():
  k = [32, 64, 128, 512]
  const = max_norm(1.0)
  init = RandomNormal(stddev=0.02)

  x0 = Input(IMG_SHAPE)

  d = Conv2D(k[0], kernel_size=3, kernel_constraint=const, strides=2, padding="same")(x0)
  d = LeakyReLU(alpha=0.2)(d)
  d = Dropout(0.25)(d)
  d = BatchNormalization(momentum=0.8)(d)

  # d = Conv2D(64, kernel_size=3, kernel_constraint=const, strides=2, padding="same")(d)
  # d = ZeroPadding2D(padding=((0,1),(0,1)))(d)
  # d = BatchNormalization(momentum=0.8)(d)
  # d = LeakyReLU(alpha=0.2)(d)

  # d = Dropout(0.25)(d)
  # d = Conv2D(128, kernel_size=3, kernel_constraint=const, strides=2, padding="same")(d)
  # d = BatchNormalization(momentum=0.8)(d)
  # d = LeakyReLU(alpha=0.2)(d)

  # d = Dropout(0.25)(d)
  # d = Conv2D(256, kernel_size=3, kernel_constraint=const, strides=1, padding="same")(d)
  # d = BatchNormalization(momentum=0.8)(d)
  # d = LeakyReLU(alpha=0.2)(d)

  # d = Dropout(0.25)(d)
  # d = Conv2D(512, kernel_size=3, kernel_constraint=const, strides=1, padding="same")(d)
  # d = BatchNormalization(momentum=0.8)(d)
  # d = LeakyReLU(alpha=0.2)(d)

  # d = Dropout(0.25)(d)
  # d = Flatten()(d)
  # d = Dense(1, activation='sigmoid')(d)

  # d = Conv2D(256, kernel_size=3, strides=2, kernel_initializer=init, padding='same')(x0)
  # d = BatchNormalization(momentum=0.8)(d)
  # d = LeakyReLU(alpha=0.2)(d)
  # d = Dropout(0.25)(d)

#   d = Conv2D(k[1], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
#   d = BatchNormalization(momentum=0.8)(d)
#   d = LeakyReLU(alpha=0.2)(d)

#   d = Conv2D(k[2], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
#   d = BatchNormalization(momentum=0.8)(d)
#   d = LeakyReLU(alpha=0.2)(d)
#   d = Dropout(0.25)(d)

#   d = Conv2D(k[3], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
#   d = BatchNormalization(momentum=0.8)(d)
#   d = LeakyReLU(alpha=0.2)(d)

  # d = Conv2D(k[3], kernel_size=3, strides=2, kernel_initializer=init, padding='same')(d)
  # d = BatchNormalization(momentum=0.8)(d)
  # d = LeakyReLU(alpha=0.2)(d)
#   d = Dropout(0.25)(d)

  # d = Concatenate()([d, yhat])(d)

  d = Flatten()(d)

#   d = Dense(1024, activation='relu')(d)
  d = Dense(1, activation='sigmoid')(d)

  d = Model(x0, d)

  d.summary()

  return d


def generator():
    #changed units
    units = IMG_SHAPE[0] * 32 * 32
    k_order = [64, 128, 256]
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    x0 = Input(shape=GEN_SHAPE)

    g = Dense(units, kernel_initializer=init, activation='relu')(x0)
    g = Reshape((32, 32, IMG_SHAPE[0]))(g)
    g = BatchNormalization(momentum=0.8)(g)

    # g = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(g)
    # g = LeakyReLU(alpha=0.2)(g)

    # g = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(g)
    # g = LeakyReLU(alpha=0.2)(g)

    for k in k_order:
        # g = UpSampling2D()(g)
        # g = Conv2D(k, kernel_size=3, kernel_initializer=init, kernel_constraint=const, strides=2, padding='same')(g)
        g = Conv2DTranspose(k, kernel_size=4, strides=2, padding='same', kernel_initializer=init)(g)
        g = BatchNormalization(momentum=0.8)(g)
        g = Activation('relu')(g)
        # g = Dropout(0.2)(g)

    #   g = Conv2D(16, kernel_size=3, padding='same')(g)
    #   g = BatchNormalization(momentum=0.8)(g)
    #   g = Activation('relu')(g)

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


# prediction path
newpath = r'C:\Users\bluet\Desktop\ImageReconstruction\predictions' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

figure_num = 0

for i in range(NUM_EPOCH):
    '''
    Train discriminator
    '''
    # idx = np.random.randint(0, yhat.shape[0]-BATCH_SIZE+1)
    # print('Random index of Discrim:', idx)
    # latent_vector = yhat[idx: idx+BATCH_SIZE]
    
    # concatenate latent features and noise
    latent_features = yhat[i: i + BATCH_SIZE]
    noise = np.random.normal(size=(BATCH_SIZE, NOISE_SIZE))
    gen_input = np.concatenate([latent_features, noise], axis=1)

    generated = g.predict(gen_input)

    # real images
    # change size
    real = X[i: i + BATCH_SIZE]

    print(real.shape, generated.shape)

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
    adv_features = yhat[i: i + BATCH_SIZE]
    noise = np.random.normal(size=(BATCH_SIZE, NOISE_SIZE))

    fake_X = np.concatenate([adv_features, noise], axis=1)
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

        plt.imshow(cv2.resize(real[0], (256, 256)))

        # for i in range(DS):
        #   for j in range(DS):
        #       axes[i, j].imshow(cv2.resize(real[count], (64, 64)))
        #       axes[i, j].axis('off')

        #       count += 1
        
        plt.savefig('predictions/real{0}.png'.format(figure_num))

        plt.imshow(cv2.resize(generated[0], (256, 256)))

        # count = 0
        # for i in range(DS):
        #     for j in range(DS):
        #         axes[i, j].imshow(cv2.resize(generated[count], (64, 64)))
        #         axes[i, j].axis('off')

        #         count += 1
                
        plt.savefig('predictions/generated{0}.png'.format(figure_num))

        figure_num += 1

            
    # plt.show()


'''
Input spectrogram + compare with original image
'''
# test_fname = os.path.join(PATH, 'n00007846', 'n00007846', 'n00007846') + '_20132.JPEG'
# print(test_fname)
# img = plt.imread(test_fname)
# img = cv2.imread(img, (64, 64))

# spect_test = os.path.join(SPECT_DIR, 'MindBigData_Imagenet_Insight_n00007846_')
# spect_test = yhat[]

# g.predict()