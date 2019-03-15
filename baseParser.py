# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:27:29 2017

@author: 1papmeie
"""


import argparse
import config_
import code
#import experiment
       
    
#Source: https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
def check_positive(value):
    """
    Checks if the given value is an int larger zero to be used in an argument parser.
    Raises a type error if not
    
    Params:
        value: input value to be checked
    Returns:
        (int) the value converted to int
    """
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive_zero(value):
    """
    Checks if the given value is an int larger or equal zero to be used in an argument parser.
    Raises a type error if not
    
    Params:
        value: input value to be checked
    Returns:
        (int) the value converted to int
    """
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def train(args,exp):
    """Sets up an experiment according to arguments from cmd parser and runs it"""
    exp.setLoader()
    exp.setAugData()
    exp.createTrainSet()
    exp.setModel()
    if args.loop:
        for i in range(5):
            exp.train()
            exp.valHoldout()
            exp.model=None
            exp.setModel()
    else:
        if args.incremental:
            exp.trainIncremental()
        else:
            exp.train()
        exp.setModel() #reload saved model with lowest val error
        exp.valHoldout()
    
def validate(args,exp):
    """Validates a model defined by arguments from cmd parser on validation set"""
    exp.setLoader()
    exp.createValidSet()
    exp.setModel()
    exp.validate(plot=args.plot)
    
def valHoldout(args,exp):
    """Validates a model defined by arguments from cmd parser on holdout set from training data"""
    exp.setLoader()
    exp.setAugData()
    exp.createTrainSet()
    exp.setModel()
    exp.valHoldout(plot=args.plot)
    
def test(args,exp):
    """Validates a model defined by arguments from cmd parser on test set"""
    exp.setLoader()
    exp.createTestSet()
    exp.setModel()
    exp.test(plot=args.plot)
    
def interactive(args,exp):
    """Sets up model defined by arguments from cmd parser and passes control to user"""
    exp.setLoader()
    exp.setAugData()
    exp.createTrainSet(shuffle=True)
    exp.createValidSet()
    exp.createTestSet()
    exp.setModel()
    code.interact(local=locals())
 

def createBaseParser():
    """Builds basic cmd parser containing common parameters for an experiment"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=check_positive_zero, help='Number of epochs, i.e. full passes over the training set', default=config_.epochs) #TODO: rename
    parser.add_argument("-b", "--batch_size", type=check_positive, help='The batch size used during training', default=config_.batchSize) #TODO: conflict NN and Mini Batch KMeans, specific parser with individual name
    parser.add_argument("-be", "--beta", type=float, help='Threshold multiplier for testing', default=config_.beta) #TODO: Move to subparsers?
    parser.add_argument("-mo", "--modelName", type=str, help='Filename of a pre-trained model to use, overwrites automatic filename interference', default=None)

    featureGroup = parser.add_argument_group('Feature extraction', 'Parameters defining dataset and feature specific values')
    featureGroup.add_argument("-ds", "--dataset", help='Which dataset to use', choices=['Marchi','DCASE2','US8K'])
    featureGroup.add_argument("-lS", "--lengthScene", type=float, help='Target length of a scene for DCASE2', default=config_.lengthScene)
    featureGroup.add_argument("-lE", "--lengthExample", type=check_positive, help='Length of an individual example presented to the model during training', default=config_.lengthExample)
    featureGroup.add_argument("-hE", "--hopExample", type=check_positive, help='Number of steps between examples', default=config_.hopExample)
    featureGroup.add_argument("-ne", "--numEvents", type=check_positive, help='Number of audio events added per artificial scene for classification training', default=config_.numEvents)
    featureGroup.add_argument("-m", "--method", help='feature extraction method', default=config_.method)
    featureGroup.add_argument("-nf", "--n_fft", type=check_positive, help='Number of bins used for the fourier transformation', default=config_.nFFT)
    featureGroup.add_argument("-hp", "--hop", type=check_positive, help='Number of frames to shift window during stft', default=config_.hop)
    featureGroup.add_argument("-nb", "--n_bin", type=check_positive, help='Number of bins used in the mel scaling', default=config_.nBin)
    featureGroup.add_argument("-p", "--prediction_delay", type=check_positive_zero, help='How many time steps forward the prediction target during training is', default=config_.predictionDelay)
    featureGroup.add_argument("-a", "--augmentation", type=int, choices=[-1,0,1,2,3,4,5], help='Which predefined data augmentation variant to use, -1 for no augmentation', default =config_.augmentation)
    featureGroup.add_argument("-lp", "--log_power", help='Turn off log power transformation for spectrogram', action=config_.logPower)
    featureGroup.add_argument("-d", "--derivate", help='Turn off first order derivate for features', action=config_.deriv)
    featureGroup.add_argument("-fe", "--frameEnergy", help='Turn off mean energy per frame for features', action=config_.frameEnergy)   
    featureGroup.add_argument("-mdB", "--max_dB", type=check_positive, help='Maximum dB threshold during feature extraction', default=config_.max_dB)
    
    parser.set_defaults(modelsFolder=config_.modelsFolder)
    parser.set_defaults(logFolder=config_.logFolder)
    parser.set_defaults(globalLog=config_.globalLog)
    parser.set_defaults(globalCsv=config_.globalCsv)
    
    return parser

def addModeParsers(parser):
    """Extends a parser with additional arguments for training modes"""
    mode_parser = parser.add_subparsers(title='Operational modes', help='Defines what experiment we want to perform on the model, train will train and validate the model') #TODO: Help description
    train_parser = mode_parser.add_parser('train')
    train_parser.add_argument("-inc", "--incremental", help="Trains in a stream like incremental fashion", action="store_true")
    train_parser.add_argument("-lo", "--loop", help="Train and evaluate a model 5 times", action="store_true")
    train_parser.set_defaults(func=train)
    valid_parser = mode_parser.add_parser('validate',aliases=['val'])
    valid_parser.add_argument("-pl", "--plot", help='Turns on plot for prediction, error and binary output', action="store_true")
    valid_parser.add_argument("-rl", "--relativeLevel", type=float, help='Scales maximum peak of added sounds in validation set relative to maximum peak of background sounds')
    valid_parser.set_defaults(func=validate)
    hold_parser = mode_parser.add_parser('validateHoldout',aliases=['valH'])
    hold_parser.add_argument("-pl", "--plot", help='Turns on plot for prediction, error and binary output', action="store_true")
    hold_parser.set_defaults(func=valHoldout)
    test_parser = mode_parser.add_parser('test')
    test_parser.add_argument("-pl", "--plot", help='Turns on plot for prediction, error and binary output', action="store_true")
    test_parser.set_defaults(func=test)
    interactive_parser = mode_parser.add_parser('interactive',aliases=['inter'])
    interactive_parser.set_defaults(func=interactive)
    return parser

if __name__ == '__main__':
    parser = createBaseParser()
    args = parser.parse_args()
    args.func(args)
    
