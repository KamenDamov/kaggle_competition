import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd


def verify_submissions(predictions):
    for file in os.listdir('submissions'):
        if file.endswith('.csv'):
            with open(os.path.join('submissions', file), 'r') as f:
                lines = f.readlines()[1:]
            label_file = np.array([int(label.split(",")[-1].strip()) for label in lines])
            if np.all(label_file == predictions):
                return file
    return None

def verify_output(predictions):
    all_files = []
    for file in os.listdir('output'):
        if file.endswith('.csv'):
            with open(os.path.join('output', file), 'r') as f:
                lines = f.readlines()[1:]
            label_file = np.array([int(label.split(",")[-1].strip()) for label in lines])
            if np.all(label_file == predictions):
                all_files.append(file)
    return all_files if len(all_files) > 0 else [None]

def save_output(predictions, classifier, params, transformations):
    today = datetime.now().strftime('%Y%m%d')
    now = datetime.now().strftime('%H%M%S')
    if not os.path.exists('output/' + today):
        os.makedirs('output/' + today)

    path = f'output/{today}/{now}_{classifier}_{params}_{transformations}.csv'
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "label"])
        for i, label in enumerate(predictions):
            writer.writerow([i, label])
    print("predictions saved in " + path)

    print("already same predictions saved in submissions?", verify_submissions(predictions))
    print("already same predictions saved in output?")
    for file in verify_output(predictions):
        print('\t', file)

    print('Number of 0:', np.sum(predictions == 0))
    print('Number of 1:', np.sum(predictions == 1))
    print('Number of differences with bayes classifier submission:', np.sum(predictions != pd.read_csv("output/output_labels_bayes_classifier.csv")['label']))



