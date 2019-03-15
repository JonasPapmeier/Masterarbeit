# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:21:06 2018

@author: brummli
"""

import baseParser
import config_
from experiment import KMeansExperiment

def addKMeansArguments(parser):
    KMeansGroup = parser.add_argument_group('KMeans', 'K-Means specific values')
    KMeansGroup.add_argument("-ncl", "--numCluster", type=baseParser.check_positive, help='Number of cluster for K-Means', default=config_.numCluster)
    KMeansGroup.add_argument("-pc", "--patience", type=baseParser.check_positive_zero, help='Number of iterations without improvement before stopping training', default=config_.patience)
    KMeansGroup.add_argument("-re", "--reassignment", type=float, help='Fraction when to reassign cluster centers', default=config_.reassignment) #TODO:description
    KMeansGroup.add_argument("-tol", "--tolerance", type=float, help='Improvement within tolerance is accepted as converged', default=config_.KMeans_tolerance)
    #KMeansGroup.add_argument("-b", "--batch_size", type=check_positive, help='The batch size used during training', default=config_.KMeans_batch_size) TODO: when we use separate files or parsers per algorithm
    
    parser.set_defaults(modelType='KMeans')

    return parser


if __name__ == '__main__':
    parser = baseParser.createBaseParser()
    kmeansParser = addKMeansArguments(parser)
    fullParser = baseParser.addModeParsers(kmeansParser)
    args = fullParser.parse_args()
    exp = KMeansExperiment(args)
    args.func(args,exp)