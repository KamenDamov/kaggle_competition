import numpy as np
from nltk import SnowballStemmer, WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import re

def compute_tf(matrix):
    """Calculer TF"""
    tf_matrix = np.zeros(matrix.shape)
    for i, doc in enumerate(matrix):
        total_terms = np.sum(doc)  
        tf_matrix[i] = doc / total_terms 
    return tf_matrix


def compute_idf(matrix):
    """Calculer Fréquence du document inverse"""
    num_docs = matrix.shape[0]
    idf = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        doc_count = np.sum(matrix[:, j] > 0)  
        idf[j] = np.log((num_docs + 1) / (1 + doc_count)) + 1  
    return idf


def compute_tfidf(tf_matrix, idf_vector):
    """Calculer TF-IDF"""
    return tf_matrix * idf_vector

def tree_based_dimensionality_reduction(X_train, y_train, top_features=5000):
    """Reduction de dimension à base d'arbre de décision"""
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    importances = tree.feature_importances_
    important_indices = np.where(importances > 0)[0]
    sorted_indices = important_indices[np.argsort(importances[important_indices])[::-1]]
    sorted_indices = sorted_indices[:top_features]
    X_train_reduced = X_train[:, sorted_indices] if isinstance(X_train, np.ndarray) else X_train.iloc[:, sorted_indices]
    
    return X_train_reduced, sorted_indices

def smote_oversampling(X, y, new_samples=1000):
    """Utilisation de l'algorithme de SMOTE pour générer de nouveaux échantillons"""
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    target_count = class_counts[minority_class] + new_samples
    smote = SMOTE(sampling_strategy={minority_class: target_count}, random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def boostrap_oversampling(X, y, new_samples=1000):
    """Utilisation de boostrap pour balancer les classes"""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    resample_indices = np.random.choice(len(X_minority), size=new_samples, replace=True)
    X_oversampled = X_minority[resample_indices]
    y_oversampled = y_minority[resample_indices]
    X_resampled = np.vstack([X, X_oversampled])
    y_resampled = np.hstack([y, y_oversampled])
    
    return X_resampled, y_resampled

def random_undersampling(X, y, new_samples=3562):
    """Sous-échantillonnage, en retirant la moitié des observations dans la classe amjoritaire"""
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y != majority_class)[0]
    np.random.seed(42)
    majority_downsampled_indices = np.random.choice(
        majority_indices, size=(len(majority_indices) - new_samples), replace=False
    )
    resampled_indices = np.concatenate([majority_downsampled_indices, minority_indices])
    np.random.shuffle(resampled_indices)
    return X[resampled_indices], y[resampled_indices]

def combine_columns(X, unique_words, word_mapping):
    """Combiner les colonnes"""
    new_X = np.zeros((X.shape[0], len(unique_words)))

    for stem, indices in word_mapping.items():
        for idx in indices:
            new_X[:, unique_words.index(stem)] += X[:, idx]

    return new_X

def get_indices_to_remove_cum_sum(all_data: np.array, threshold: float) -> np.array:
    """Retirer les mots qui reviennent rarement par le biais de la somme cumulative"""
    feature_sums = np.sum(all_data, axis=0)
    sorted_indices = np.argsort(feature_sums)[::-1]
    sorted_sums = feature_sums[sorted_indices]
    cum_sum = np.cumsum(sorted_sums)
    total_sum = cum_sum[-1]
    cutoff_index = np.searchsorted(cum_sum, total_sum * threshold)
    indices_to_keep = sorted_indices[:cutoff_index + 1]
    mask = np.ones(feature_sums.shape, dtype=bool)
    mask[indices_to_keep] = False
    indices_to_remove = np.where(mask)[0]
    return indices_to_remove

class DataPreprocess:
    """Classe pour lancer les méthodes de prétraitement"""
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

        self.train = np.array(self.train, dtype=np.float32)
        self.test = np.array(self.test, dtype=np.float32)


    def initialize_tfidf(self):
        self.train_tfidf = compute_tfidf(compute_tf(self.train), compute_idf(self.train))
        self.test_tfidf = compute_tfidf(compute_tf(self.test), compute_idf(self.test))

    def remove_cum_sum(self):
        indices_to_remove = get_indices_to_remove_cum_sum(self.train, 0.95)
        self.train = np.delete(self.train, indices_to_remove, axis=1)
        self.test = np.delete(self.test, indices_to_remove, axis=1)

    def remove_stopwords(self):
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

    def apply_stemming(self):
        stemmer = SnowballStemmer("english")
        stemmed_word_mapping = {}
        for idx, word in enumerate(self.vocab_map):
            stemmed_word = stemmer.stem(word)
            if stemmed_word not in stemmed_word_mapping:
                stemmed_word_mapping[stemmed_word] = []
            stemmed_word_mapping[stemmed_word].append(idx)

        # Create a new combined matrix
        # The number of unique stemmed words will determine the new matrix shape
        unique_stems = list(stemmed_word_mapping.keys())

        self.train = combine_columns(self.train, unique_stems, stemmed_word_mapping)
        self.test = combine_columns(self.test, unique_stems, stemmed_word_mapping)
        self.vocab_map = np.array(unique_stems)

    def apply_lemmatization(self):
        # Initialize the WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Create a mapping of lemmatized words to their original indices
        lemmatized_word_mapping = {}
        for idx, word in enumerate(self.vocab_map):
            lemmatized_word = lemmatizer.lemmatize(word)  # specify the POS if needed
            if lemmatized_word not in lemmatized_word_mapping:
                lemmatized_word_mapping[lemmatized_word] = []
            lemmatized_word_mapping[lemmatized_word].append(idx)

        # Create a new combined matrix for lemmatized words
        unique_lemmas = list(lemmatized_word_mapping.keys())

        self.train = combine_columns(self.train, unique_lemmas, lemmatized_word_mapping)
        self.test = combine_columns(self.test, unique_lemmas, lemmatized_word_mapping)
        self.vocab_map = np.array(unique_lemmas)


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





