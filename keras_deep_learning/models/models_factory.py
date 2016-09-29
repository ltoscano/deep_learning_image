'''
Created on Sep 29, 2016

@author: dearj019
'''
from keras_deep_learning.models import vgg16

def preprocess_factory(model_type):
    
    if model_type == "vgg16":
        return vgg16.preprocess
    else:
        #defaults to vgg16
        return vgg16.preprocess

def model_forward_factory(model_type, regr, no_labels):
    if model_type == "vgg16":
        return vgg16.model_forward(regr, no_labels)
    else:
        #defaults to vgg16
        return vgg16.model_forward(regr, no_labels)
    pass

def model_factory(model_type, lr, regr, no_labels):
    if model_type == "vgg16":
        return vgg16.model(lr, regr, no_labels)
    else:
        #Default to vgg16
        return vgg16.model(lr, regr, no_labels)