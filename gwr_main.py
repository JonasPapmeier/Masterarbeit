# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:26:04 2018

@author: 1papmeie
"""

import baseParser
import config_
from experiment import GWRExperiment

def addGWRArguments(parser):
    GWRGroup = parser.add_argument_group('GWR', 'Growing When Required Network specific values')
    GWRGroup.add_argument("-mno", "--maxNodes", type=baseParser.check_positive, help="The maximum number of Nodes allowed in the network",default=config_.maxNodes)
    GWRGroup.add_argument("-mne", "--maxNeighbours", type=baseParser.check_positive, help="The maximum number of neighbours a node can have",default=config_.maxNeighbours)
    GWRGroup.add_argument("-ma", "--maxAge", type=baseParser.check_positive, help="The maximum age of an edge before it gets deleted",default=config_.maxAge)
    GWRGroup.add_argument("-ht", "--habituationThreshold", type=float, help='Bla', default=config_.habThres) 
    GWRGroup.add_argument("-it", "--insertThreshold", type=float, help='What distance has to be exceed at minimum before adding a new node', default=config_.insThres)
    GWRGroup.add_argument("-eb", "--epsilonB", type=float, help='Bla', default=config_.epsilonB) 
    GWRGroup.add_argument("-en", "--epsilonN", type=float, help='Bla', default=config_.epsilonN)
    GWRGroup.add_argument("-tb", "--tauB", type=float, help="tau B",default=config_.tauB)
    GWRGroup.add_argument("-tn", "--tauN", type=float, help="tau N",default=config_.tauN)
    
    parser.set_defaults(modelType='GWR')

    return parser


if __name__ == '__main__':
    parser = baseParser.createBaseParser()
    gwrParser = addGWRArguments(parser)
    fullParser = baseParser.addModeParsers(gwrParser)
    args = fullParser.parse_args()
    exp = GWRExperiment(args)
    args.func(args,exp)