from keras.layers import merge, Deconvolution2D, UpSampling2D, Convolution2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, LeakyReLU
from keras.models import Model
import os

nBottleneck = 25
batchSz = 32

imDim = 256

imageInput = Input(shape=(imDim, imDim, 3))

def makeEncoder():
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
