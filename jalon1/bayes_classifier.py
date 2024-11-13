import numpy as np
from preprocess_data import DataPreprocess
import csv
from tqdm import tqdm

from save_output import save_output


class BayesClassifier:
    def __init__(self) -> None:
        self.priors = None
        self.likelihoods = None
        self.classes = None
    
    def compute_prior(self, X_train, y_train):
        """Calcule les probabilités à priori de chaque classe pour chaque document"""
        number_of_docs = len(X_train)
        priors = np.array([np.sum(y_train == cls) / number_of_docs for cls in self.classes])
        return priors

    def compute_likelihoods(self, X_train: np.array, y_train: np.array, laplace_smoothing: int):
        """Calcule les vraisemblances pour chaque classe pour chaque document"""
        likelihoods = []
        vocab_size = X_train.shape[-1]
        for cls in self.classes: 
            mask = y_train == cls
            class_filter = X_train[mask]
            total_words = np.sum(np.sum(class_filter, axis=1))
            word_counts = np.sum(class_filter, axis=0) + laplace_smoothing
            likelihoods.append((word_counts) / (total_words + laplace_smoothing * vocab_size))
        return likelihoods

    def predict(self, X_new: np.array): 
        """Prédit les classes des nouvelles données"""
        log_probs = []
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = np.log(self.likelihoods[cls])
            log_prob = prior + np.sum(X_new * likelihood)
            log_probs.append(log_prob)
        return self.classes[np.argmax(log_probs)]

    def random_split(self, X: np.array, y: np.array, split_ratio: float = 0.9):
        """Sépare de façon aléatoire le set de données X et y en deux sets de données selon le split_ratio donné"""
        indices = np.random.permutation(len(X))
        split_index = int(len(X) * split_ratio)
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def fit(self, X: np.array, y: np.array, laplace_smoothing: float = 1.0):
        """Entraine le modèle"""
        self.classes = np.unique(y)
        self.priors = self.compute_prior(X, y)
        self.likelihoods = self.compute_likelihoods(X, y, laplace_smoothing)

    def f1_score(self, y_true, y_pred):
        """Calcule le f1_score"""
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def hyperparameter_tuning(self, X, y):
        """Entraine plusieurs modèles avec une recherche par grille et retourne l'hyperparamètre qui a obtenu le
        meilleur score f1"""
        laplace_smoothers = np.arange(0.4, 1.05, 0.05)
        max_f1 = 0
        best_lps = 0
        for lps in tqdm(laplace_smoothers):
            f1_scores = []
            for k in range(7): 
                X_train, y_train, X_val, y_val = self.random_split(X, y)
                self.fit(X_train, y_train, lps)
                predictions = np.array([self.predict(x_i) for x_i in X_val])
                f1 = self.f1_score(y_val, predictions)
                f1_scores.append(f1)
            f1_mean = np.mean(f1_scores)
            if f1_mean > max_f1: 
                best_lps = lps
                max_f1 = f1_mean 
        return best_lps, max_f1
            

if __name__ == "__main__": 
    data_preprocess = DataPreprocess()
    print("data processed")

    bayes_classifier = BayesClassifier()
    best_lps, best_f1 = bayes_classifier.hyperparameter_tuning(data_preprocess.train, data_preprocess.label_train)
    print(best_lps, best_f1)

    tuned_bayes_classifier = BayesClassifier()
    tuned_bayes_classifier.fit(data_preprocess.train, data_preprocess.label_train)
    predictions = np.array([tuned_bayes_classifier.predict(x_i) for x_i in data_preprocess.test])

    save_output(predictions, "bayes", best_lps, "stopwords")
