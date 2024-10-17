import numpy as np


class DataPreprocess: 
    test: np.array = np.array([]) 
    train: np.array = np.array([])
    vocab_map: np.array = np.array([])
    label_train: np.array = list
    train_tfidf: np.array = np.array([])
    def __init__(self) -> None:
        self.test = np.load('./classer-le-text/data_test.npy', allow_pickle=True)
        self.train = np.load('./classer-le-text/data_train.npy', allow_pickle=True)
        self.vocab_map = np.load('./classer-le-text/vocab_map.npy', allow_pickle=True)
        with open('./classer-le-text/label_train.csv', 'r') as file:
            lines = file.readlines()[1:]
        self.label_train = np.array([int(label.split(",")[-1].strip()) for label in lines])
        self.train_tfidf = self.compute_tfidf(self.compute_tf(self.train), self.compute_idf(self.train))

    # Step 1: Compute Term Frequency (TF)
    def compute_tf(self, matrix):
        tf_matrix = np.zeros(matrix.shape)
        for i, doc in enumerate(matrix):
            total_terms = np.sum(doc)  # Total number of terms in document
            tf_matrix[i] = doc / total_terms  # TF for each term
        return tf_matrix

    # Step 2: Compute Inverse Document Frequency (IDF)
    def compute_idf(self, matrix):
        num_docs = matrix.shape[0]
        idf = np.zeros(matrix.shape[1])
        for j in range(matrix.shape[1]):
            doc_count = np.sum(matrix[:, j] > 0)  # Number of documents containing the term
            idf[j] = np.log((num_docs + 1) / (1 + doc_count)) + 1  # Smoothing
        return idf

    # Step 3: Compute TF-IDF
    def compute_tfidf(self, tf_matrix, idf_vector):
        return tf_matrix * idf_vector


if __name__ == "__main__":
    dataPrepocess = DataPreprocess()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)
