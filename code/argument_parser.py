# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 02:27:54 2019

@author: Preety
"""

import argparse

def classifier_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--type', type=str, default='fc',
                        help='select type of classifier [\'fc\', \'svm\'] (default: fc)')
    
    args = parser.parse_args()
    return args

def feature_extractor_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', type=str, help='absolute path to data input directory', required=True)
    
    args = parser.parse_args()
    return args