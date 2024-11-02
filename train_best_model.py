from xgboost import XGBClassifier
from preprocess_data import DataPreprocess

from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pickle
import csv
with open('results_2.pkl', 'rb') as file:
    data = pickle.load(file)
idx = np.argmax(np.array(data['xgboost']['f1_scores']))
best_params = np.array(data['xgboost']['hyperparameters'])[idx]
best_params