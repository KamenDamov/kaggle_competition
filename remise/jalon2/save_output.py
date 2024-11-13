import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd


def verify_submissions(predictions):
    """Vérifie si une soumission a les mêmes prédictions"""
    for file in os.listdir('submissions'):
        if file.endswith('.csv'):
            with open(os.path.join('submissions', file), 'r') as f:
                lines = f.readlines()[1:]
            label_file = np.array([int(label.split(",")[-1].strip()) for label in lines])
            if np.all(label_file == predictions):
                return file
    return None


def verify_output(predictions, newly_saved):
    """Vérifie si une ou plusieurs anciennes prédictions ont les mêmes prédictions"""
    all_files = []
    for root, dirs, files in os.walk('output'):
        for file in files:
            if file.endswith('.csv') and os.path.join(root, file) != newly_saved:
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()[1:]
                label_file = np.array([int(label.split(",")[-1].strip()) for label in lines])
                if np.all(label_file == predictions):
                    all_files.append(os.path.join(root, file))
    return all_files if len(all_files) > 0 else [None]


def save_output(predictions, classifier, params, transformations):
    """Sauve les prédictions dans un dossier du jour (sous ./output) avec le temps à laquelle il a été enregistré et quelle a été
    le classifieur utilisé, les paramètres finaux utilisés et les transformations appliquées"""

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
    for file in verify_output(predictions, path):
        print('\t', file)

    print('Number of 0:', np.sum(predictions == 0))
    print('Number of 1:', np.sum(predictions == 1))
    print('Ratio of 1:', np.sum(predictions == 1) / len(predictions))

    bayes_submission = pd.read_csv("output/output_labels_bayes_classifier.csv")['label']
    print('Number of differences with bayes classifier submission:', np.sum(predictions != bayes_submission))
    print('Ratio of 1 in bayes submission:', np.sum(bayes_submission == 1) / len(bayes_submission))
