# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:58:33 2019

@author: Preety
"""

import argument_parser

import os
import shutil
import random

args = argument_parser.feature_extractor_args()
input_path = args.input_path

random.seed(42)

def split_data(base_path, input_path, class_names):
    train_dir = base_path + "train/"
    test_dir = base_path + "test/"

    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    os.makedirs(base_path, mode=0o777)

    os.makedirs(train_dir, mode=0o777)
    os.makedirs(test_dir, mode=0o777)
    
    for dir in class_names:
        if not os.path.exists(train_dir + dir):
            os.mkdir(train_dir + dir, mode=0o777)
        if not os.path.exists(test_dir + dir):
            os.mkdir(test_dir + dir, mode=0o777)
        current_dir = input_path + dir
        files = os.listdir(current_dir)
        random.shuffle(files)
        
        test_split_idx = int(len(files) * 0.30) 
        files_for_test = files[:test_split_idx]
        files_for_train = files[test_split_idx:]
        
        for file in files_for_train:
            shutil.copy2(os.path.join(current_dir, file),
                         os.path.join(train_dir, dir))
        for file in files_for_test:
            shutil.copy2(os.path.join(current_dir, file),
                         os.path.join(test_dir, dir))


def main():
    
    class_names = ["commercial_area", "river", "circular_farmland", "snowberg", "rectangular_farmland", "forest",
                   "industrial_area", "terrace", "mountain", "dense_residential", "desert",
                   "medium_residential", "lake", "meadow", "sparse_residential", "wetland"]
    
    base_path = os.path.join(os.getcwd(), "/dataset/")
    split_data(base_path, input_path, class_names)


if __name__ == '__main__':
    main()