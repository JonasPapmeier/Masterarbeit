# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:21:06 2018

@author: brummli
"""

import baseParser
import config_
from experiment import NNExperiment

def addNNArguments(parser):
    nnGroup = parser.add_argument_group('NN', 'Neural Net specific values')
    nnGroup.add_argument("-l", "--layers", type=baseParser.check_positive_zero, help='Number of layers in the neural net model', default=config_.layers) 
    nnGroup.add_argument("-hid", "--hiddenNodes", type=baseParser.check_positive, help='Number of hidden Units in each layer', default=config_.hiddenNodes)
    nnGroup.add_argument("-lr", "--learningRate", type=float, help='Initial learning rate', default=config_.lr)
    nnGroup.add_argument("-s", "--sigma", type=float, help='Width of gaussian noise added for denoising autoencoder', default=config_.sigma)
    
    parser.add_argument("modelType", help='defines the model type',choices=['RNN_AE','LSTM','FF','FF_AE','SEQ_AE','SEQ'])

    return parser


if __name__ == '__main__':
    parser = baseParser.createBaseParser()
    nnParser = addNNArguments(parser)
    fullParser = baseParser.addModeParsers(nnParser)
    args = fullParser.parse_args()
    exp = NNExperiment(args)
    args.func(args,exp)