from keras.layers import merge, Deconvolution2D, UpSampling2D, Convolution2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, LeakyReLU
from keras.models import Model
import os

nBottleneck = 25
batchSz = 32

imDim = 256

# first, define the encoder
imageInput = Input(shape=(imDim, imDim, 3))

# imDim x imDim input
x = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', dim_ordering='tf')(imageInput)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 128 x 128 input
x = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 64 x 64 input
x = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 32 x 32 input
x = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 16 x 16 input
x = Convolution2D(128, 4, 4, subsample=(2, 2), border_mode='same', dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 8 x 8 input
x = Convolution2D(128, 4, 4, subsample=(2, 2), border_mode='same', dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 4 x 4 input
encodedFeatures = Convolution2D(nBottleneck, 4, 4, subsample=(1, 1), border_mode='valid', dim_ordering='tf')(x)

encoder_model = Model(imageInput, encodedFeatures, name='Encoder')
encoder_model.summary()

# decoder_model 
encoded_input = Input(shape=(1,1,nBottleneck))

# 1,1,2000 input
x = UpSampling2D(size=(4,4),dim_ordering='tf')(encoded_input)
x = Convolution2D(128, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 4x4x512 input
x = Convolution2D(128, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D(size=(2,2),dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 8x8ximDim input
x = Convolution2D(64, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D(size=(2,2),dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 16x16ximDim input
x = Convolution2D(64, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D(size=(2,2),dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 32x32x128 input
x = Convolution2D(64, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D(size=(2,2),dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 64x64x64 input
x = Convolution2D(32, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D(size=(2,2),dim_ordering='tf')(x)
x = BatchNormalization(mode=2)(x)
x = LeakyReLU(alpha=0.2)(x)

# 128x128x64 input
x = Convolution2D(3, 4, 4, subsample=(1, 1), border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D(size=(2,2),dim_ordering='tf')(x)

generator_model = Model(encoded_input,x,name="Decoder")
generator_model.summary()

x = encoder_model(imageInput)
x = generator_model(x)
autoencoder = Model(imageInput, x)
autoencoder.summary()

from keras.preprocessing.image import ImageDataGenerator
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import image

## Let's test some learning rates
import numpy as np
from numpy import random
import tqdm
from IPython.core.debugger import Pdb


numEpochs = 50
numBatchesPerEpoch = 1000
nValBatches = 100

trainPath = "/home/ubuntu/SoftRobotRepresentation/Data/Training/0/"
valPath = "/home/ubuntu/SoftRobotRepresentation/Data/Validation/0/"

def trainGenerator():
    files = os.listdir(trainPath)
    files = files[:numBatchesPerEpoch*batchSz]
    while True:
        res = np.zeros(shape=(batchSz,imDim,imDim,3))
        for i in range(numBatchesPerEpoch):
            for j in range(batchSz):
                res[j,:] = plt.imread(trainPath+files[i*batchSz+j]).astype('float32')/255.0
            yield (res,res)
        
def valGenerator():  
    files = os.listdir(valPath)
    files = files[:nValBatches*batchSz]
    res = np.zeros(shape=(batchSz,imDim,imDim,3))
    while True:
        for i in range(nValBatches):
            for j in range(batchSz):
                res[j,:] = plt.imread(valPath+files[i*batchSz+j]).astype('float32')/255.0
            yield (res,res)

## Let's test some learning rates
from keras.optimizers import Adam
import numpy as np
from numpy import random
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


lr = 0.05
print "Learning rate=" + str(lr)
print "Loading Weights"
autoencoder.load_weights('/home/ubuntu/SoftRobotRepresentation/ModelPrettyGoodBottleneck25.h5')
print "Done Loading Weights"
print "Compiling model"
autoencoder.compile(optimizer=Adam(lr=lr),loss='mse')
print "Done compiling model"
history = autoencoder.fit_generator(generator=trainGenerator(),
                                    samples_per_epoch=numBatchesPerEpoch*batchSz,
                                    nb_epoch=numEpochs,
                                    validation_data=valGenerator(),
                                    nb_val_samples=nValBatches*batchSz,
                                    callbacks=[TensorBoard(log_dir='/home/ubuntu/SoftRobotRepresentation/Data/Logs/3'),
                                               ModelCheckpoint('/home/ubuntu/SoftRobotRepresentation/Model.h5', save_best_only=True),
                                               ReduceLROnPlateau(factor=0.5,patience=5,verbose=True)]
                                   )
