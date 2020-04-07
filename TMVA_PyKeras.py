from __future__ import print_function
import ROOT
from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Reshape


# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()


output = TFile.Open('TMVA_PyKeras_model.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=None:AnalysisType=Classification')


## Loading the data file
inputFileName = "/Users/gollum/Downloads/sample_images_32x32.root"
inputFile = ROOT.TFile.Open(inputFileName)

signalTree     = inputFile.Get("sig_tree")
backgroundTree = inputFile.Get("bkg_tree")

dataloader = TMVA.DataLoader('dataset')

dataloader.AddSignalTree(signalTree, 1.0)
dataloader.AddBackgroundTree(backgroundTree, 1.0)

# Add variables
imgSize = 32 * 32
for i in range(imgSize):
    varName = "var_{} := vars[{}]".format(i,i)
    dataloader.AddVariable(varName,'F')

dataloader.PrepareTrainingAndTestTree( ROOT.TCut(''), ROOT.TCut(''),
                                  "nTrain_Signal=8000:nTrain_Background=8000:SplitMode=Random:"
                                   "NormMode=NumEvents:!V" )

    
model = Sequential()
model.add(Reshape((32, 32, 1), input_shape=(imgSize,)))
model.add(Conv2D(6, (5, 5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Store model to file
model.save('TMVA_PyKeras_model.h5')
model.summary()


# Book method
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=None:FilenameModel=TMVA_PyKeras_model.h5:'
                   'FileNameTrainedModel=trained_model_cnn.h5:NumEpochs=10:BatchSize=256')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

roc = factory.GetROCCurve(dataloader)
roc.SaveAs('TMVA_PyKeras_ROC.png')