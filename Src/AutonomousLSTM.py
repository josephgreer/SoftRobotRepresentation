%matplotlib inline
timeSz = 32
from MakeModels import *
encoder = makeEncoder(timeSz)
decoder = makeDecoder()
autoencoder = makeAutoencoder(encoder,decoder)

autoencoder.load_weights('/home/ubuntu/SoftRobotRepresentation/ModelPrettyGoodBottleneck25.h5', by_name=True)

imageLSTM = makeImageLSTM(encoder)

## Let's test some learning rates
import numpy as np
from numpy import random
import tqdm
from IPython.core.debugger import Pdb
from matplotlib import pyplot as plt

import pandas as pd


numEpochs = 50
numBatchesPerEpoch = 1000
nValBatches = 100

trainPath = "/home/ubuntu/SoftRobotRepresentation/Data/Training/0/"
valPath = "/home/ubuntu/SoftRobotRepresentation/Data/Validation/0/"

positionFiles = ['/home/ubuntu/SoftRobotRepresentation/Data/recordingRight.csv',
                 '/home/ubuntu/SoftRobotRepresentation/Data/recordingRight2.csv']

numFramesTrain = [21420,31430]
numFramesVal = [9180,13470]
numFrames = [e+f for e,f in zip(numFramesTrain,numFramesVal)]

predictAhead = 5

posTables = [pd.read_csv(e) for e in positionFiles]

positionDataTrain = np.empty(shape=(0,2))
positionDataVal = np.empty(shape=(0,2))
for i in range(len(positionFiles)):
    positionData = posTables[i].as_matrix(columns=['x(m)','y(m)'])
    positionDataTrain = np.concatenate((positionDataTrain,positionData[:numFramesTrain[i],:]))
    positionDataVal = np.concatenate((positionDataVal,positionData[numFramesTrain[i]:numFrames[i],:]))
    
print str(positionDataTrain.shape) +" " + str(positionDataVal.shape)

def trainGenerator():
    files = os.listdir(trainPath)
    files = ['%s%06d.jpg'%(trainPath,i) for i in range(len(files))]
    res = np.zeros(shape=(timeSz,imDim,imDim,3))
    while True:
        for i in range(numBatchesPerEpoch):
            resPos = positionDataTrain[i*timeSz:(i+1)*(timeSz),:]
            for j in range(timeSz):
                index = i*timeSz+j
                res[j,:] = plt.imread(files[index]).astype('float32')/255.0
                
            posRes = positionDataTrain[i*timeSz+predictAhead:(i+1)*timeSz+predictAhead,:]
            yield ([resPos,res],posRes)
        
def valGenerator():  
    files = os.listdir(valPath)
    files = ['%s%06d.jpg'%(valPath,i) for i in range(len(files))]
    res = np.zeros(shape=(timeSz,imDim,imDim,3))
    while True:
        for i in range(nValBatches):
            resPos = positionDataVal[i*timeSz:(i+1)*(timeSz),:]
            for j in range(timeSz):
                index = i*timeSz+j
                res[j,:] = plt.imread(files[index]).astype('float32')/255.0
                
            posRes = positionDataVal[i*timeSz+predictAhead:(i+1)*timeSz+predictAhead,:]
            yield ([resPos,res],posRes)
            
            

def trainGeneratorPlainPos():
    res = np.zeros(shape=(timeSz,2))
    while True:
        for i in range(numBatchesPerEpoch):
            res = positionDataTrain[i*timeSz:i*(timeSz+1),:]
            posRes = positionDataTrain[i*(timeSz+1)+predictAhead,:]
            yield (res,posRes)
        
def valGenerator():  
    res = np.zeros(shape=(timeSz,2))
    while True:
        for i in range(nValBatches):
            res = positionDataVal[i*timeSz:i*(timeSz+1),:]
            posRes = positionDataVal[i*(timeSz+1)+predictAhead,:]
            yield (res,posRes)

## Let's test some learning rates
from keras.optimizers import Adam
import numpy as np
from numpy import random
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


lr = 5e-2
print "Learning rate=" + str(lr)
print "Compiling model"
imageLSTM.compile(optimizer=Adam(lr=lr),loss='mse')
print "Done compiling model"
history = imageLSTM.fit_generator(generator=trainGenerator(),
                                    samples_per_epoch=numBatchesPerEpoch*timeSz,
                                    nb_epoch=numEpochs,
                                    validation_data=valGenerator(),
                                    nb_val_samples=nValBatches*timeSz,
                                    callbacks=[TensorBoard(log_dir='/home/ubuntu/SoftRobotRepresentation/Data/Logs/0'),
                                               ModelCheckpoint('/home/ubuntu/SoftRobotRepresentation/LSTMModel.h5'),
                                               ReduceLROnPlateau(factor=0.2,patience=2)]
                                   )
