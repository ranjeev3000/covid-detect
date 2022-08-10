#MAIN FLASK FILE:
from flask import Flask, render_template, request, send_from_directory
import os
import pickle
import skimage
import sklearn
import pandas as pd
import xgboost as xgb
from sklearn import svm
from sklearn import tree
from skimage import feature
from lime import lime_image
from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime.wrappers.scikit_image import SegmentationAlgorithm
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
from hog_capture import *

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) 
    import lime

UPLOAD_FOLDER = '.\\static\\images\\uploads'

#Dataset Extraction using paths:

# #Covid Positive path: 
# positive_path = 'E:/Thesis/Dataset/Dataset_Kaggle/COVID-19_Radiography_Dataset/COVID'

# #Covid Negative path:
# negative_path = 'E:/Thesis/Dataset/Dataset_Kaggle/COVID-19_Radiography_Dataset/Normal'

#Loading all the ML models using pickle:

# model = pickle.load(open("Pickle/Pickle Files/randomforest_edit.pkl", "rb"))
# model1 = pickle.load(open("svm_edit.pkl", "rb"))
# model2 = pickle.load(open("xgboost_edit.pkl", "rb"))
# model3 = pickle.load(open("decisiontree_edit.pkl", "rb"))

#Using Flask API:

app = Flask(__name__)

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static\\images\\upload')

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method=='POST':
        image = request.files['input-image']
        input_file_name = secure_filename(image.filename)
        input_path = os.path.join(UPLOADS_PATH, input_file_name)
        image.save(input_path)
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        hog_img = request.form.get('input-image')
        get_hog(input_path)
        return(render_template("projectHOG.html", input_path =input_file_name))
    return render_template("projectHOG.html")


##Included new till here
if (__name__ == "__main__"):
    app.run(port=3000, debug=True)