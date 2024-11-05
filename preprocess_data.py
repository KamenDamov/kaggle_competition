# import numpy as nnp
import numpy as np
from nltk import WordNetLemmatizer

# import truncated_svd
from sklearn.decomposition import PCA, TruncatedSVD
from nltk.stem import SnowballStemmer

# import cupy as np
nnp = np
# import pca


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


class DataPreprocess:
    test: np.array = np.array([]) 
    train: np.array = np.array([])
    vocab_map: np.array = np.array([])
    label_train: np.array = list
    def __init__(self) -> None:
        self.test = np.load('./classer-le-text/data_test.npy', allow_pickle=True).astype(np.float32)
        self.train = np.load('./classer-le-text/data_train.npy', allow_pickle=True).astype(np.float32)
        self.vocab_map = nnp.load('./classer-le-text/vocab_map.npy', allow_pickle=True)
        with open('./classer-le-text/label_train.csv', 'r') as file:
            lines = file.readlines()[1:]
        self.label_train = np.array([int(label.split(",")[-1].strip()) for label in lines])

    def initialize_tfidf(self):
        self.train = compute_tfidf(compute_tf(self.train), compute_idf(self.train))
        self.test= compute_tfidf(compute_tf(self.test), compute_idf(self.test))

    def remove_min_max(self, min, max):
        # min or max can be absolute value or % value
        if 0 < max < 1:
            max_tfidf_scores = np.max(self.train, axis=0)
            percentile_threshold = np.percentile(max_tfidf_scores, max*100)
            features_to_keep = max_tfidf_scores <= percentile_threshold
            self.train = self.train[:,features_to_keep]
            self.test = self.test[:,features_to_keep]
            # self.vocab_map = self.vocab_map[features_to_keep.get()]
            self.vocab_map = self.vocab_map[features_to_keep]
        elif max > 0:
            max_tfidf_scores = np.max(self.train, axis=0)
            idx_to_delete = np.argsort(max_tfidf_scores)[-max:]
            # self.vocab_map = nnp.delete(self.vocab_map, idx_to_delete, axis=0)
            self.vocab_map = nnp.delete(self.vocab_map, idx_to_delete.get(), axis=0)
            self.train = np.delete(self.train, idx_to_delete, axis=1)
            self.test = np.delete(self.test, idx_to_delete, axis=1)
        if 0 < min < 1:
            min_tfidf_scores = np.min(self.train, axis=0)
            percentile_threshold = np.percentile(min_tfidf_scores, min * 100)
            features_to_keep = min_tfidf_scores <= percentile_threshold
            self.train = self.train[:, features_to_keep]
            self.test = self.test[:, features_to_keep]
            # self.vocab_map = self.vocab_map[features_to_keep.get()]
            self.vocab_map = self.vocab_map[features_to_keep]
        elif min > 0:
            min_tfidf_scores = np.min(self.train, axis=0)
            idx_to_delete = np.argsort(min_tfidf_scores)[-min:]
            # self.vocab_map = nnp.delete(self.vocab_map, idx_to_delete.get(), axis=0)
            self.vocab_map = nnp.delete(self.vocab_map, idx_to_delete, axis=0)
            self.train = np.delete(self.train, idx_to_delete, axis=1)
            self.test = np.delete(self.test, idx_to_delete, axis=1)

    def remove_stopwords(self):
        # TODO: use list instead of opening document
        with open('english_stopwords', 'r') as f:
            stopwords = [line.strip() for line in f.readlines()]
        idx_stopwords = [i for i in range(len(self.vocab_map)) if self.vocab_map[i] in stopwords and np.sum(self.train[:,i]) > 10000]
        self.vocab_map = nnp.delete(self.vocab_map, idx_stopwords, axis=0)
        self.train = np.delete(self.train, idx_stopwords, axis=1)
        self.test = np.delete(self.test, idx_stopwords, axis=1)

    def apply_stemming(self):
        stemmer = SnowballStemmer("english")

        # Create a mapping of stemmed words to their original indices
        stemmed_word_mapping = {}
        for idx, word in enumerate(self.vocab_map):
            stemmed_word = stemmer.stem(word)
            if stemmed_word not in stemmed_word_mapping:
                stemmed_word_mapping[stemmed_word] = []
            stemmed_word_mapping[stemmed_word].append(idx)

        # Create a new combined matrix
        # The number of unique stemmed words will determine the new matrix shape
        unique_stems = list(stemmed_word_mapping.keys())
        new_train = np.zeros((self.train.shape[0], len(unique_stems)))
        new_test = np.zeros((self.test.shape[0], len(unique_stems)))

        # Fill the new matrix with summed occurrences
        for stem, indices in stemmed_word_mapping.items():
            for idx in indices:
                new_train[:, unique_stems.index(stem)] += self.train[:, idx]
                new_test[:, unique_stems.index(stem)] += self.test[:, idx]

        self.train = new_train
        self.test = new_test
        self.vocab_map = np.array(unique_stems)

    def apply_lemmatization(self):
        # Initialize the WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Create a mapping of lemmatized words to their original indices
        lemmatized_word_mapping = {}
        for idx, word in enumerate(self.vocab_map):
            lemmatized_word = lemmatizer.lemmatize(word)  # You may specify the POS if needed
            if lemmatized_word not in lemmatized_word_mapping:
                lemmatized_word_mapping[lemmatized_word] = []
            lemmatized_word_mapping[lemmatized_word].append(idx)

        # Create a new combined matrix for lemmatized words
        unique_lemmas = list(lemmatized_word_mapping.keys())
        new_train = np.zeros((self.train.shape[0], len(unique_lemmas)))
        new_test = np.zeros((self.test.shape[0], len(unique_lemmas)))

        # Fill the new matrix with summed occurrences
        for lemma, indices in lemmatized_word_mapping.items():
            for idx in indices:
                new_train[:, unique_lemmas.index(lemma)] += self.train[:, idx]
                new_test[:, unique_lemmas.index(lemma)] += self.test[:, idx]

        self.train = new_train
        self.test = new_test
        self.vocab_map = np.array(unique_lemmas)

    def apply_pca(self, n_components):
        # self.train, _, _ = pca.incremental_pca(self.train, 500, n_components)
        # self.test, _, _ = pca.incremental_pca(self.test, 500, n_components)
        pca = PCA(n_components=10000)

        self.train = pca.fit_transform(self.train)
        self.test = pca.fit_transform(self.test)

    def apply_truncated_svd(self, n_components):
        truncated_svd = TruncatedSVD(n_components=n_components, algorithm='arpack')
        self.train = truncated_svd.fit_transform(self.train)
        self.test = truncated_svd.fit_transform(self.test)


if __name__ == "__main__":
    dataPrepocess = DataPreprocess()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)

    # dataPrepocess.remove_stopwords()
    # print('train', dataPrepocess.train.shape)
    # print('test', dataPrepocess.test.shape)
    # print('vocab_map', dataPrepocess.vocab_map.shape)
    # print('label_train', dataPrepocess.label_train.shape)

    # dataPrepocess.apply_truncated_svd(10000)
    # print('train', dataPrepocess.train.shape)
    # print('test', dataPrepocess.test.shape)
    # print('vocab_map', dataPrepocess.vocab_map.shape)
    # print('label_train', dataPrepocess.label_train.shape)

    # dataPrepocess.apply_truncated_svd(2000)
    # print('train', dataPrepocess.train.shape)
    # print('test', dataPrepocess.test.shape)

    # dataPrepocess.initialize_tfidf()
    # dataPrepocess.remove_min_max(5, 0.999)
    # print('train', dataPrepocess.train.shape)
    # print('test', dataPrepocess.test.shape)

    dataPrepocess.apply_stemming()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)

    dataPrepocess.apply_lemmatization()
    print('train', dataPrepocess.train.shape)
    print('test', dataPrepocess.test.shape)
    print('vocab_map', dataPrepocess.vocab_map.shape)
    print('label_train', dataPrepocess.label_train.shape)
