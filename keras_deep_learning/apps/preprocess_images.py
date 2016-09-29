'''
This script pre-process a folder of images and converts 
them into numpy arrays to be ready for input for
vgg-16 models type

Created on Sep 27, 2016

@author: dearj019
'''

import imageio
import sys
import json
import os
import numpy as np
from keras_deep_learning.models.models_factory import preprocess_factory

def preprocess_images(images_folder, processed_output_folder, model_type):
    
    labels = [d for d in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, d))]
    labels.sort()
    preprocess = preprocess_factory(model_type)
      
    for i in range(len(labels)):
        label = labels[i]
        images = [str(a) for a in os.listdir(os.path.join(images_folder, label)) if os.path.isfile(os.path.join(images_folder, label, a))]
        if not os.path.exists(os.path.join(processed_output_folder, label)):
            os.makedirs(os.path.join(processed_output_folder, label))
        
        for image in images:
            image_path = os.path.join(images_folder, label, image)
            try:
                im = imageio.imread(image_path)
            except:
                print(image_path)
                continue
            
            im = preprocess(im)
            if im == None:
                print(image_path)
                continue
                    
            np.save(open(os.path.join(processed_output_folder, label, image + ".npy"), "w+"), im)
    
    
def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    images_folder = conf['images_folder']
    processed_output_folder = conf['processed_output_folder']
    model_type = conf['model_type']
    preprocess_images(images_folder, processed_output_folder, model_type)
    
if __name__ == "__main__":
    main()