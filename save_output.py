import csv
import os
from datetime import date
import numpy as np


def verify_predictions(predictions):
    for file in os.listdir('submissions'):
        if file.endswith('.csv'):
            with open(os.path.join('output', file), 'r') as f:
                lines = f.readlines()[1:]
            label_file = np.array([int(label.split(",")[-1].strip()) for label in lines])
            if label_file == predictions:
                return file
    return False

def save_output(predictions, classifier, params, transformations):
    today = date.today().strftime('%Y%m%d')
    if not os.path.exists('output/' + today):
        os.makedirs('output/' + today)

    path = f'output/{today}/{classifier}_{params}_{transformations}.csv'
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "label"])
        for i, label in enumerate(predictions):
            writer.writerow([i, label])
    print("predictions saved in " + path)

    print("already same predictions saved in submissions?", verify_predictions(predictions))


