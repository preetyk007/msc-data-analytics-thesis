# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 01:10:26 2019

@author: Preety
"""

import split_data
import argument_parser

import os
import h5py
import glob
import shutil
import numpy as np

from keras import optimizers
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet152, preprocess_input

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

args = argument_parser.feature_extractor_args()
input_path = args.input_path

image_size = (224, 224)

base_path = os.path.join(os.getcwd(), "./dataset/")
output_path = os.path.join(os.getcwd(), "./output/")

train_dir = base_path + "train/"
test_dir = base_path + "test/"


class_names = ["commercial_area", "river", "circular_farmland", "snowberg", "rectangular_farmland", "forest",
               "industrial_area", "terrace", "mountain", "dense_residential", "desert",
               "medium_residential", "lake", "meadow", "sparse_residential", "wetland"]

def build_cnn_model():
    base_model = ResNet152((224,224,3), weights="imagenet")
    cnn_model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
    
    opt = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
    cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return cnn_model


def image_preprocessing(path, mode='train'):
    features = []
    labels = []
    
    train_datagen = ImageDataGenerator(
                                    rotation_range=40,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    for i, class_name in enumerate(class_names):
        class_path = path + class_name
        count = 1
        for image_path in glob.glob(class_path + "/*.jpg"):
            # Load and resize image
            im = image.load_img(image_path, target_size=image_size)
            # Convert to numpy array
            x = image.img_to_array(im)
            # Expand dimensions for further preprocessing 
            x = np.expand_dims(x, axis=0)
            # Data augmentation
            aug_cnt = 0
            if( mode == 'test' ):
                pass
            else:
                datagen = train_datagen
                for batch in datagen.flow(x, batch_size=1):
                    batch = preprocess_input(batch)
                    features.append(batch)
                    labels.append(class_name)
                    aug_cnt += 1
                    if aug_cnt >= 1:
                        break
            x = preprocess_input(x)
            features.append(x)
            labels.append(class_name)
            count += 1
            print("[INFO] completed image - " + image_path)
        print("[INFO] completed label - " + class_name)
        
    return features, labels


def bottleneck_feature_extractor(data, le, model):
    
    features = []
    labels = []
    
    print("started bottleneck feature extraction")
    for label, img in zip(le, data):
        feature = model.predict(img)
        flt_feature = feature.flatten()
        features.append(flt_feature)
        labels.append(label)
    print("finished bottleneck feature extraction")
    
    # One-hot encoding for labels
    le = LabelEncoder()
    le_labels = le.fit_transform(labels)
    le_labels = np_utils.to_categorical(le_labels, len(class_names))
        
    # Data array
    le_labels = np.array(le_labels)
    features = np.array(features)
    
    return features, le_labels


def save_features_labels(mode, features, labels):
    features_path = output_path + mode + "_features_resnet152" + ".h5"
    labels_path = output_path + mode + "_labels_resnet152" + ".h5"
    
    h5f_data = h5py.File(features_path, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(features))
    
    h5f_label = h5py.File(labels_path, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(labels))
    
    h5f_data.close()
    h5f_label.close()


def main():
    split_data.split_data(base_path, input_path, class_names)
    print("[INFO] completed data splitting ")
    
    train_data, train_labels = image_preprocessing(train_dir)
    test_data, test_labels = image_preprocessing(test_dir, 'test')
    print("[INFO] completed data preprocessing ")

    model = build_cnn_model()
    print("[INFO] completed model building ")

    X_train, Y_train = bottleneck_feature_extractor(train_data, train_labels, model)
    X_test, Y_test = bottleneck_feature_extractor(test_data, test_labels, model)
    print("[INFO] completed bottleneck feature extraction ")
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    os.makedirs(output_path, mode=0o777)
    
    save_features_labels('train', X_train, Y_train)
    save_features_labels('test', X_test, Y_test)
    print("[INFO] completed saving bottleneck features ")


if __name__ == '__main__':
    main()