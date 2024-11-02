import numpy as np
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import re

def compute_tf(matrix):
    tf_matrix = np.zeros(matrix.shape)
    for i, doc in enumerate(matrix):
        total_terms = np.sum(doc)  # Total number of terms in document
        tf_matrix[i] = doc / total_terms  # TF for each term
    return tf_matrix


def compute_idf(matrix):
    num_docs = matrix.shape[0]
    idf = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        doc_count = np.sum(matrix[:, j] > 0)  # Number of documents containing the term
        idf[j] = np.log((num_docs + 1) / (1 + doc_count)) + 1  # Smoothing
    return idf


def compute_tfidf(tf_matrix, idf_vector):
    return tf_matrix * idf_vector

def tree_based_dimensionality_reduction(X_train, y_train, threshold=0.000001, max_features=None):
    # Initialize the Decision Tree Classifier
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    importances = tree.feature_importances_
    important_indices = np.where(importances >= threshold)[0]
    sorted_indices = important_indices[np.argsort(importances[important_indices])[::-1]]
    if max_features:
        sorted_indices = sorted_indices[:max_features]
    X_train_reduced = X_train[:, sorted_indices] if isinstance(X_train, np.ndarray) else X_train.iloc[:, sorted_indices]
    
    return X_train_reduced, sorted_indices

def smote_oversampling(X, y, new_samples=1500):
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    target_count = class_counts[minority_class] + new_samples
    smote = SMOTE(sampling_strategy={minority_class: target_count}, random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled


class DataPreprocess:
    test: np.array = np.array([]) 
    train: np.array = np.array([])
    vocab_map: np.array = np.array([])
    label_train: np.array = list
    train_tfidf: np.array = np.array([])
    test_tfidf: np.array = np.array([])
    def __init__(self) -> None:
        self.test = np.load('./classer-le-text/data_test.npy', allow_pickle=True)
        self.train = np.load('./classer-le-text/data_train.npy', allow_pickle=True)
        self.vocab_map = np.load('./classer-le-text/vocab_map.npy', allow_pickle=True)
        with open('./classer-le-text/label_train.csv', 'r') as file:
            lines = file.readlines()[1:]
        self.label_train = np.array([int(label.split(",")[-1].strip()) for label in lines])

    def initialize_tfidf(self):
        self.train_tfidf = compute_tfidf(compute_tf(self.train), compute_idf(self.train))
        self.test_tfidf = compute_tfidf(compute_tf(self.test), compute_idf(self.test))

    def remove_min_max(self, min, max):
        # min or max can be absolute value or % value
        pass

    def remove_stopwords(self):
        # TODO: use list instead of opening document
        with open('english_stopwords', 'r') as f:
            stopwords = [line.strip() for line in f.readlines()]
        idx_stopwords = [i for i in range(len(self.vocab_map)) if self.vocab_map[i] in stopwords]
        self.vocab_map = np.delete(self.vocab_map, idx_stopwords, axis=0)
        self.train = np.delete(self.train, idx_stopwords, axis=1)
        self.test = np.delete(self.test, idx_stopwords, axis=1)

    def remove_non_words(self):
        with open('english_stopwords', 'r') as f:
            stopwords = [line.strip() for line in f.readlines()]
        idx_stopwords = [i for i in range(len(self.vocab_map)) if self.vocab_map[i] in stopwords]
        self.vocab_map = np.delete(self.vocab_map, idx_stopwords, axis=0)
        self.train = np.delete(self.train, idx_stopwords, axis=1)
        self.test = np.delete(self.test, idx_stopwords, axis=1)
        
        filtered_indices = [index for index, token in enumerate(self.vocab_map) if not re.match(r'^[a-zA-Z]{2,}$', token)]
        self.vocab_map = np.delete(self.vocab_map, filtered_indices, axis=0)
        self.train = np.delete(self.train, filtered_indices, axis=1)
        self.test = np.delete(self.test, filtered_indices, axis=1)


if __name__ == "__main__":
    dataPrepocess = DataPreprocess()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)

    dataPrepocess.remove_stopwords()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)

    # document_term_matrix = np.array([
    #     [3, 0, 1, 0],  # Document 1
    #     [1, 1, 0, 1],  # Document 2
    #     [0, 2, 0, 3]  # Document 3
    # ])
    # tf_matrix = compute_tf(document_term_matrix)
    # idf_vector = compute_idf(document_term_matrix)
    # tfidf_matrix = compute_tfidf(tf_matrix, idf_vector)
    #
    # print("TF-IDF Matrix:")
    # print(tfidf_matrix)





