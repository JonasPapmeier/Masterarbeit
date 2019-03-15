# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:21:06 2018

@author: brummli
"""

import baseParser
import config_
from experiment import AutoencoderSmallNNExperiment
import nn_main


def addExtraNNClassifierArguments(parser):
    
    parser.add_argument("-nle", "--NNpred_lengthExample", type=baseParser.check_positive, help='The number of frames to use', default=1) #TODO: change default
    parser.add_argument("-nhe", "--NNpred_hopExample", type=baseParser.check_positive, help='The number of frames to advance per example', default=1) #TODO: change default
    parser.add_argument("-nl", "--NNpred_layers",type=baseParser.check_positive_zero,help='The number of layers for the neural net classifier',default=0)
    parser.add_argument("-nhid", "--NNpred_hiddenNodes",type=baseParser.check_positive,help='The number of hidden nodes per layer if hidden layers are used',default=50)
    parser.add_argument("-nlr", "--NNpred_lr",type=float,help='The learning rate for the neural net classifier',default=0.1)
    parser.add_argument("-nmo", "--AE_ModelName", type=str, help='Specifies the feature extraction neural net by its file name', default=None)
    parser.add_argument("-el", "--extractionLayer", type=baseParser.check_positive_zero, help='Which layers output of the autoencoder will be used to extract a transformed representation', default=config_.extractionLayer)
    
    parser.set_defaults(modelType='NN')

    return parser


if __name__ == '__main__':
    parser = baseParser.createBaseParser()
    nnParser = nn_main.addNNArguments(parser)
    combinedParser = addExtraNNClassifierArguments(nnParser)
    fullParser = baseParser.addModeParsers(combinedParser)
    args = fullParser.parse_args()
    exp = AutoencoderSmallNNExperiment(args)
    args.func(args,exp)