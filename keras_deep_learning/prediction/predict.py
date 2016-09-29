'''
Created on Sep 28, 2016

@author: dearj019
'''

import sys
import json
import os
from keras_deep_learning.models.models_factory import model_forward_factory
from keras_deep_learning.models.models_factory import preprocess_factory

class Classifier():
    def __init__(self, model_type, weights_path, regr, labels):
        self.__forward_model = model_forward_factory(model_type, regr, len(labels))
        self.__forward_model.load_weights(weights_path)
        self.__preprocess_function = preprocess_factory(model_type)
        self.__labels = labels
        self.__labels.sort()
         
    def classify(self, im):
        im = self.__preprocess_function(im)
        if im is None:
            return {}
        
        p = self.__forward_model.predict(im)
        to_return = {}
        for i in range(p.shape[1]):
            value = p[0, i]
            to_return[self.__labels[i]] = value
        return to_return
    
    def close_session(self):
        pass