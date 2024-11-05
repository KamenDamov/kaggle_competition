import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from preprocess_data import DataPreprocess
import csv
import tqdm

from save_output import save_output
from visualize_data import plot_grid_search


class BayesClassifier:
    def __init__(self) -> None:
        self.priors = None
        self.likelihoods = None
        self.classes = None
    
    def compute_prior(self, X_train, y_train):
        number_of_docs = len(X_train)
        priors = np.array([np.sum(y_train == cls) / number_of_docs for cls in self.classes])
        return priors

    def compute_likelihoods(self, X_train: np.array, y_train: np.array, laplace_smoothing: int): 
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
        log_probs = []
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = np.log(self.likelihoods[cls])
            log_prob = prior + np.sum(X_new * likelihood)
            log_probs.append(log_prob)
        return self.classes[np.argmax(log_probs)]

    def random_split(self, X: np.array, y: np.array, split_ratio: float = 0.9):
        indices = np.random.permutation(len(X))
        split_index = int(len(X) * split_ratio)
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def fit(self, X: np.array, y: np.array, laplace_smoothing: float = 1.0):
        self.classes = np.unique(y)
        self.priors = self.compute_prior(X, y)
        self.likelihoods = self.compute_likelihoods(X, y, laplace_smoothing)

    def f1_score(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def hyperparameter_tuning(self, X, y):
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


def grid_search(X, y):
    alpha = np.arange(0.01, 1, 0.01)
    param_grid = dict(alpha=alpha)
    print(param_grid)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    grid = GridSearchCV(ComplementNB(), param_grid=param_grid, cv=cv, scoring='f1_weighted', verbose=3)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


if __name__ == "__main__": 
    data_preprocess = DataPreprocess()
    print("data processed")
    data_preprocess.remove_stopwords()
    print("stopwords removed")
    data_preprocess.initialize_tfidf()
    print("tf-idf processed")
    # data_preprocess.apply_truncated_svd(100)
    # print('truncated svd applied!')
    data_preprocess.remove_min_max(0.01, 0.999)
    print("min-max removed")

    train = data_preprocess.train
    label_train = data_preprocess.label_train
    test = data_preprocess.test
    print('train:', train.shape)
    print('test:', test.shape)
    print('Ratio of 1 in train:', np.sum(label_train == 1) / len(label_train))

    # bayes_classifier = BayesClassifier()
    # best_lps, best_f1 = bayes_classifier.hyperparameter_tuning(data_preprocess.train, data_preprocess.label_train)
    # print(best_lps, best_f1)
    #
    # tuned_bayes_classifier = BayesClassifier()
    # tuned_bayes_classifier.fit(data_preprocess.train, data_preprocess.label_train)
    # predictions = np.array([tuned_bayes_classifier.predict(x_i) for x_i in data_preprocess.test])

    best_params_, best_score_ = grid_search(train, label_train)
    print('best params: {}, score: {}'.format(best_params_, best_score_))

    bayes_classifier = ComplementNB(alpha=best_params_['alpha'])
    bayes_classifier.fit(train, label_train)
    print('f1_score on train:', f1_score(label_train, bayes_classifier.predict(train)))

    y_pred = bayes_classifier.predict(test)

    params = f'alpha={bayes_classifier.alpha}'
    save_output(y_pred, "comp_bayes", params, "stopwords_tf-idf_min-max")
