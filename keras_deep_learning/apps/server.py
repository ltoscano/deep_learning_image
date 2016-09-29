'''
Created on Sep 29, 2016

@author: dearj019
'''

from keras_deep_learning.prediction.predict import Classifier
from flask import Flask, jsonify, request
import json
import signal
import sys
import imageio

app = Flask(__name__)
app.secret_key = "super secret key"

classifier = None

def signal_handler(signal, frame):
    if classifier != None:
        classifier.close_session()
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

@app.route('/characters/classify', methods = ['POST'])
def classify():
    if request.method == 'POST':
        binary = request.stream.read()
        im = imageio.imread(binary)
        results = classifier.classify(im)
        return str(results)

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    model_path = conf['model_path']
    labels_file = conf['labels_file']
    port = conf['port']
    model_type = conf['model_type']
    with open(labels_file) as f:
        labels = f.read().strip().split("\n")
    
    classifier = Classifier(model_type, model_path, None, labels)
    app.run(host = '0.0.0.0', port = port)
        
    