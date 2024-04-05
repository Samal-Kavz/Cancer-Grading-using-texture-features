from joblib import load
import pandas as pd
from datetime import datetime
import os
#data = pd.read_csv('features.csv')
def predict(dataset):
    #data = pd.read_csv(dataset)
    #names = data['name']
    #data = data.drop(columns=['name'])
    #names=['0.jpg']
    pipeline = load('svm.joblib')
    grades = pipeline.predict(dataset)
    probs = pd.DataFrame(pipeline.predict_proba(dataset)[0],columns=['probs'])
    probs['class'] = pipeline.classes_
    sorted_probs = probs.sort_values(['probs'],ascending=False)
    sorted_probs = sorted_probs.reset_index(drop=True)
    return sorted_probs
    
