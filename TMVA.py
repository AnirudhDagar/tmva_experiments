import ROOT
from ROOT import TMVA
import os 

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()


outputFile = ROOT.TFile.Open("TMVA_Test_CNN_Output.root", "RECREATE")
factory = ROOT.TMVA.Factory("TMVA_CNN_Classification", outputFile,
                      "!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" )

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

#CNN output_size = (input_size - filter_size + 2 * padding)/stride + 1


inputLayoutString = "InputLayout=1|32|32"
                                                                                                
batchLayoutString = "BatchLayout=256|1|1024"
                                                   

layoutString = ("Layout=CONV|6|5|5|1|1|0|0|RELU,MAXPOOL|2|2|1|1,CONV|16|5|5|1|1|0|0|RELU,MAXPOOL|2|2|1|1,"
            "RESHAPE|FLAT,DENSE|120|RELU,DENSE|84|RELU,DENSE|1|LINEAR")

##Training strategies.                                                                                                                          
t1 = ("LearningRate=0.001,Repetitions=1,"
                     "ConvergenceSteps=10,BatchSize=256,TestRepetitions=1,"
                     "MaxEpochs=10,Regularization=None,"
                     "Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.0")
                                                                          
 
trainingStrategyString = "TrainingStrategy=" + t1
    
cnnOptions = ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"
                       "WeightInitialization=XAVIERUNIFORM");

cnnOptions +=  ":" + inputLayoutString
cnnOptions +=  ":" + batchLayoutString
cnnOptions +=  ":" + layoutString
cnnOptions +=  ":" + trainingStrategyString
cnnOptions +=  ":Architecture=CPU"


factory.BookMethod(dataloader, TMVA.Types.kDL, "DL_CNN", cnnOptions)

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

roc = factory.GetROCCurve(dataloader)
roc.SaveAs('TMVA_ROC.png')
