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
UPLOADS_ML = join(dirname(realpath(__file__)), 'static\\images\\ML')
UPLOADS_DL = join(dirname(realpath(__file__)), 'static\\images\\DL')

@app.route('/',methods=['GET','POST'])
def hog():
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

@app.route('/ml',methods=['GET','POST'])
def ml_pred():
    global input_file_name
    global lime_path
    if request.method=='POST':

        if request.form['action'] == 'SVM':
            try:
                image = request.files['input-image']
                input_file_name = secure_filename(image.filename)
                input_path = os.path.join(UPLOADS_ML, input_file_name)
                image.save(input_path)
            except Exception as e:
                print(e)
            print(input_file_name)
            print('SVG')
            prediction_proba = "0.5"
            prediction_text = "prediction SVM"
        elif request.form['action'] == 'Random Forest':
            print('Random Forest')
            prediction_proba = "0.6"
            prediction_text = "prediction RF"
        elif request.form['action'] == 'XG Boost':
            print('XG Boost')
            prediction_proba = "0.7"
            prediction_text = "prediction XG Boost"
        elif request.form['action'] == 'LIME XAI-SVM':
            print('LIME XAI-SVM')
            prediction_proba = "0.8"
            prediction_text = "prediction LIME XAI-SVM"
            lime_path = "../static/images/Lime/Lime image showing positive and negative regions.png"
            return(render_template("projectML.html",input_path =input_file_name, lime_path=lime_path))

        elif request.form['action'] == 'LIME XAI-Random Forest':
            print('LIME XAI-Random Forest')
            prediction_proba = "0.9"
            prediction_text = "prediction LIME XAI-Random Forest"
            return(render_template("projectML.html",input_path =input_file_name, lime_path=lime_path))
        elif request.form['action'] == 'LIME XAI-XG Boost':
            print('LIME XAI-XG Boost')
            prediction_proba = "1.0x"
            prediction_text = "prediction LIME XAI-XG Boost"
        # image = request.files['input-image']
        # input_file_name = secure_filename(image.filename)
        # input_path = os.path.join(UPLOADS_PATH, input_file_name)
        # image.save(input_path)
            return(render_template("projectML.html", input_path =input_file_name, lime_path=lime_path))
        return(render_template("projectML.html", prediction_proba=prediction_proba, prediction_text=prediction_text, input_path =input_file_name))
    return render_template("projectML.html")

@app.route('/dl',methods=['GET','POST'])
def dl_pred():
    global input_file_dl
    lime_path = "../static/images/Lime/Lime image showing positive regions.png"
    print('dl page')
    if request.method == 'POST':
        if request.form['action'] == 'ExceptionNet B3':
            try:
                image = request.files['input-image']
                input_file_dl = secure_filename(image.filename)
                input_path = os.path.join(UPLOADS_DL, input_file_dl)
                image.save(input_path)
            except Exception as e:
                print(e)
            print(input_file_dl)
            print('Exception Net')
            prediction_proba = "0.5"
            prediction_text = "prediction Exception net"
        if request.form['action'] == 'LIME XAI':
            if request.files['input-image']:
                print('image path needed')
                try:
                    image = request.files['input-image']
                    input_file_dl = secure_filename(image.filename)
                    input_path = os.path.join(UPLOADS_DL, input_file_dl)
                    image.save(input_path)
                except Exception as e:
                    print(e)
                print(input_file_dl)
                print('Exception Net')
                prediction_proba = "0.5"
                prediction_text = "prediction LIME IM"
            else:
                print("No imagpath needed")
                prediction_proba = "0.6"
                prediction_text = "prediction lIME"
            return(render_template("projectDL.html", input_path =input_file_dl, lime_path=lime_path))

        return(render_template("projectDL.html", prediction_proba=prediction_proba, prediction_text=prediction_text, input_path =input_file_dl))
    return render_template("projectDL.html")

##Included new till here
if (__name__ == "__main__"):
    app.run(port=3000, debug=True)