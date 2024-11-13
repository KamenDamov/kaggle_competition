import numpy as np
from nltk import SnowballStemmer, WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import re


class DataPreprocess:
    test: np.array = np.array([]) 
    train: np.array = np.array([])
    vocab_map: np.array = np.array([])
    label_train: np.array = list
    train_tfidf: np.array = np.array([])
    test_tfidf: np.array = np.array([])
    def __init__(self) -> None:
        """Initialise les données"""
        self.test = np.load('./classer-le-text/data_test.npy', allow_pickle=True)
        self.train = np.load('./classer-le-text/data_train.npy', allow_pickle=True)
        self.vocab_map = np.load('./classer-le-text/vocab_map.npy', allow_pickle=True)
        with open('./classer-le-text/label_train.csv', 'r') as file:
            lines = file.readlines()[1:]
        self.label_train = np.array([int(label.split(",")[-1].strip()) for label in lines])

        self.train = np.array(self.train, dtype=np.int8)  # convertit en int8 pour accélérer les calculs
        self.test = np.array(self.test, dtype=np.int8)  # convertit en int8 pour accélérer les calculs


if __name__ == "__main__":
    dataPrepocess = DataPreprocess()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)
