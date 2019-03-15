# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:21:06 2018

@author: brummli
"""

import baseParser
import config_
from experiment import AutoencoderGWRExperiment
import nn_main
import gwr_main


def addExtraGWRArguments(parser):
    parser.add_argument("-gle", "--GWR_lengthExample", type=baseParser.check_positive, help='The number of frames to use', default=1) #TODO: change default
    parser.add_argument("-ghe", "--GWR_hopExample", type=baseParser.check_positive, help='The number of frames to advance per example', default=1) #TODO: change default
    parser.add_argument("-gmo", "--GWR_NNModelName", type=str, help='Specifies the feature extraction neural net by its file name', default=None)
    parser.add_argument("-el", "--extractionLayer", type=baseParser.check_positive_zero, help='Which layers output of the autoencoder will be used to extract a transformed representation', default=config_.extractionLayer)
    
    parser.set_defaults(modelType='GWR')

    return parser


if __name__ == '__main__':
    parser = baseParser.createBaseParser()
    nnParser = nn_main.addNNArguments(parser)
    gwrParser = gwr_main.addGWRArguments(nnParser)
    combinedParser = addExtraGWRArguments(gwrParser)
    fullParser = baseParser.addModeParsers(combinedParser)
    args = fullParser.parse_args()
    exp = AutoencoderGWRExperiment(args)
    args.func(args,exp)