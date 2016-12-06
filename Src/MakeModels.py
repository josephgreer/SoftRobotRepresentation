from keras.layers import Lambda,merge, Deconvolution2D, UpSampling2D, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Merge, BatchNormalization, LeakyReLU, LSTM
from keras.models import Model
from keras import backend as K
import os

nBottleneck = 25

imDim = 256

imageInput = Input(shape=(imDim, imDim, 3))

nWindow = 1
positionInput = Input(shape=(2*nWindow,))
nHiddenLSTM = 64

def makeEncoder(batchSz):
    gBatchSz = batchSz
    # first, define the encoder

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
    return encoder_model

def makeDecoder():
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
    return generator_model

def makeAutoencoder(encoder_model, generator_model):
    x = encoder_model(imageInput)
    x = generator_model(x)
    autoencoder = Model(imageInput, x)
    return autoencoder

def makePlainLSTM():
    x = LSTM(nHiddenLSTM)(positionInput)
    x = Dense(round(nHiddenLSTM/2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(2)(x)
    plainLSTM = Model(positionInput,x)
    return plainLSTM

def squeezeTwice(x):
    x = K.squeeze(x,1)
    return K.squeeze(x,1)

def expandDims(x):
    return K.expand_dims(x,dim=0)

def makeImageLSTM(encoder):
    y = encoder(imageInput)
    y = Lambda(squeezeTwice)(y)
    #y = Flatten()(y)
    print K.int_shape(y)
    x = merge([positionInput,y],mode='concat')
    x = Lambda(expandDims)(x)
    print K.int_shape(x)
    x = LSTM(nHiddenLSTM)(x)
    print K.int_shape(x)
    x = Dense(round(nHiddenLSTM/2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(2)(x)
    print K.int_shape(x)
    imageLSTM = Model([positionInput,imageInput],x)
    return imageLSTM
