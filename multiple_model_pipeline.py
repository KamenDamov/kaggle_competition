import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import preprocess_data

from preprocess_data import DataPreprocess
from scipy.stats import uniform, randint, loguniform
import random
import pickle
from sklearn.metrics import confusion_matrix
import time
import warnings
import csv

# Suppress all warnings
warnings.filterwarnings('ignore', category=UserWarning)
class ClassifierPipeline:
    best_model_xgboost: XGBClassifier
    best_model_logistic: LogisticRegression
    best_naive_bayes: MultinomialNB
    def __init__(self):
        pass

    def fit_best(self, X: np.array, y: np.array, params:dict, best_model: BaseEstimator) -> None:
        if isinstance(best_model, XGBClassifier):
            model = best_model(**params)
            model.fit(X, y)
            self.best_model_xgboost = model
        elif isinstance(best_model, LogisticRegression):
            model = best_model(**params)
            model.fit(X, y)
            self.best_model_logistic = model
        elif isinstance(best_model, MultinomialNB):
            model = best_model(**params)
            model.fit(X, y)
            self.best_naive_bayes = model
        else:
            raise ValueError("Unsupported model type")

    def sample_hyperparameters(self, param_distributions):
        sampled_params = {}
        for param, distribution in param_distributions.items():
            if isinstance(distribution, list):
                sampled_params[param] = random.choice(distribution)
            else:
                sampled_params[param] = distribution.rvs()
        return sampled_params

    def hyperparameter_tuning(self, X: np.array, y: np.array, n_iter=20):
        param_xgboost = {
            'n_estimators': randint(100, 300),
            'max_depth': randint(3, 11),
            'learning_rate': uniform(0.01, 0.29),
            'min_child_weight': randint(1, 11),
            'colsample_bytree': uniform(0.5, 0.5),
        }

        # Define hyperparameter search space for Logistic Regression
        param_logistic = {
            'C': loguniform(0.01, 10),
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l2'],
        }

        param_bayes = {
            'alpha': uniform(0.01, 1.2),
        }
        param_complement_bayes = {
            'alpha': uniform(0.01, 1.2),
            'norm': [True, False]
        }

        f1_scorer = make_scorer(f1_score, average='binary')

        f1_scores_xgboost = []
        f1_scores_logistic = []
        f1_scores_bayes = []
        f_1_scores_complement_bayes = []

        hyperparams_xgboost = []
        hyperparams_logistic = []
        hyperparams_bayes = []
        hyperparams_complement_bayes = []

        confusion_matrices_xgboost = []
        confusion_matrices_logistic = []
        confusion_matrices_bayes = []
        confusion_matrices_complement_bayes = []
        
        for i in range(n_iter):
            print(f"Hyperparameter iteration {i+1}/{n_iter}")
            print("Current best XGBoost", max(f1_scores_xgboost) if f1_scores_xgboost != [] else 0)
            print("Current best Logistic", max(f1_scores_logistic) if f1_scores_logistic != [] else 0)
            print("Current best Naive Bayes", max(f1_scores_bayes) if f1_scores_bayes != [] else 0)
            print("Current best Complement Naive Bayes", max(f_1_scores_complement_bayes) if f_1_scores_complement_bayes != [] else 0)

            sampled_hyperparams_xgboost = self.sample_hyperparameters(param_xgboost)
            sampled_hyperparams_logistic = self.sample_hyperparameters(param_logistic)
            sampled_hyperparams_bayes = self.sample_hyperparameters(param_bayes)
            sampled_hyperparams_complement_bayes = self.sample_hyperparameters(param_complement_bayes)
            
            print("Params xgboost: ", str(sampled_hyperparams_xgboost))
            print("Params logistic: ", str(sampled_hyperparams_logistic))
            print("Params bayes: ", str(sampled_hyperparams_bayes))
            print("Params complement bayes: ", str(sampled_hyperparams_complement_bayes))

            f1_scores_complement_bayes_fold = []
            f1_scores_xgboost_fold = []
            f1_scores_logistic_fold = []
            f1_scores_bayes_fold = []

            confusion_matrices_complement_bayes_fold = []
            confusion_matrices_xgboost_fold = []
            confusion_matrices_logistic_fold = []
            confusion_matrices_bayes_fold = []
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

            # Apply cross-validation
            for train_index, val_index in skf.split(X, y):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                #X_train_tfidf = tfidf_transformer.transform(X_train).astype(np.float32)
                #X_val_tfidf = tfidf_transformer.transform(X_val)

                # Apply bootstrap resampling to the TF-IDF-transformed training data
                #X_train_resampled, y_train_resampled = preprocess_data.apply_bootstrap_resampling(X_train_tfidf, y_train)

                # Train XGBoost
                model_xgboost = XGBClassifier(**sampled_hyperparams_xgboost)
                model_xgboost.fit(X_train, y_train)
                y_pred_xgboost = model_xgboost.predict(X_val)
                f1_scores_xgboost_fold.append(f1_scorer(model_xgboost, X_val, y_val))
                confusion_matrices_xgboost_fold.append(confusion_matrix(y_val, y_pred_xgboost))

                # Train Logistic Regression
                model_logistic = LogisticRegression(**sampled_hyperparams_logistic, max_iter=1000)
                model_logistic.fit(X_train, y_train)
                y_pred_logistic = model_logistic.predict(X_val)
                f1_scores_logistic_fold.append(f1_scorer(model_logistic, X_val, y_val))
                confusion_matrices_logistic_fold.append(confusion_matrix(y_val, y_pred_logistic))

                model_bayes = MultinomialNB()
                model_bayes.fit(X_train, y_train)
                y_pred_bayes = model_bayes.predict(X_val)
                f1_scores_bayes_fold.append(f1_scorer(model_bayes, X_val, y_val))
                confusion_matrices_bayes_fold.append(confusion_matrix(y_val, y_pred_bayes))

                model_complement_bayes = ComplementNB(**sampled_hyperparams_complement_bayes)
                model_complement_bayes.fit(X_train, y_train)
                y_pred_complement_bayes = model_complement_bayes.predict(X_val)
                f1_scores_complement_bayes_fold.append(f1_scorer(model_complement_bayes, X_val, y_val))
                confusion_matrices_complement_bayes_fold.append(confusion_matrix(y_val, y_pred_complement_bayes))


            # Store results for each hyperparameter configuration
            f1_scores_xgboost.append(np.mean(f1_scores_xgboost_fold))
            f1_scores_logistic.append(np.mean(f1_scores_logistic_fold))
            f1_scores_bayes.append(np.mean(f1_scores_bayes_fold))
            f_1_scores_complement_bayes.append(np.mean(f1_scores_complement_bayes_fold))

            hyperparams_xgboost.append(sampled_hyperparams_xgboost)
            hyperparams_logistic.append(sampled_hyperparams_logistic)
            hyperparams_bayes.append(sampled_hyperparams_bayes)
            hyperparams_complement_bayes.append(sampled_hyperparams_complement_bayes)

            confusion_matrices_xgboost.append(confusion_matrices_xgboost_fold)
            confusion_matrices_logistic.append(confusion_matrices_logistic_fold)
            confusion_matrices_bayes.append(confusion_matrices_bayes_fold)
            confusion_matrices_complement_bayes.append(confusion_matrices_complement_bayes_fold)

        avg_f1_xgboost = np.mean(f1_scores_xgboost)
        avg_f1_logistic = np.mean(f1_scores_logistic)
        avg_f1_bayes = np.mean(f1_scores_bayes)
        avg_f1_complement_bayes = np.mean(f_1_scores_complement_bayes)

        # Store the F1-scores, hyperparameters, and confusion matrices for each model
        self.results = {
            'xgboost': {
                'f1_scores': f1_scores_xgboost,
                'hyperparameters': hyperparams_xgboost,
                'confusion_matrices': confusion_matrices_xgboost
            },
            'logistic_regression': {
                'f1_scores': f1_scores_logistic,
                'hyperparameters': hyperparams_logistic,
                'confusion_matrices': confusion_matrices_logistic
            },
            'bayes': {
                'f1_scores': f1_scores_bayes,
                'hyperparameters': hyperparams_bayes,
                'confusion_matrices': confusion_matrices_bayes
            },
            'complement_bayes': {
                'f1_scores': f_1_scores_complement_bayes,
                'hyperparameters': hyperparams_bayes,
                'confusion_matrices': confusion_matrices_bayes
            }
        }

        return {
            'avg_f1_xgboost': avg_f1_xgboost,
            'avg_f1_logistic': avg_f1_logistic,
            'avg_f1_bayes': avg_f1_bayes,
            'avg_f1_complement_bayes': avg_f1_complement_bayes
        }
        

    def get_results(self):
        with open("results_multiple_models_no_tfidf.pkl", 'wb') as file:
            pickle.dump(self.results, file)
        print(f"Results saved")
        return self.results

cp = ClassifierPipeline()
data_preprocess = DataPreprocess()
print("Data preprocessed")

# Step 1: Apply TF-IDF and Mutual Information Gain on Training Data
print("Reducing dimensionality")
start = time.time()

# Apply TF-IDF transformation to the entire training set
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(data_preprocess.train).astype(np.float32)

# Perform mutual information gain dimensionality reduction
#top_features_indices = preprocess_data.mutual_info_gain_dimensionality_reduction(X_train_tfidf, data_preprocess.label_train)
#X_train_reduced = data_preprocess.train[:, top_features_indices]
#print("Dimensionality reduced in ", str(time.time() - start))
#np.save('reduced.npy', X_train_tfidf)

# Step 2: Hyperparameter Tuning
print("Tuning hyperparameters")
data_preprocess.remove_stopwords()
tuning_results = cp.hyperparameter_tuning(data_preprocess.train, data_preprocess.label_train)
print("Tuning done")
results = cp.get_results()

# Load results for model selection
with open('results_multiple_modelsno_tfidf.pkl', 'rb') as file:
    data = pickle.load(file)

# Step 3: Apply Dimensionality Reduction to Test Data
#X_test_tfidf = tfidf_transformer.transform(data_preprocess.test).astype(np.float32)
#X_test_reduced = X_test_tfidf[:, top_features_indices]

# Step 4: Training Best Models and Generating Predictions

# Best Logistic Regression Model
idx = np.argmax(np.array(data['logistic_regression']['f1_scores']))
best_params_logistic = data['logistic_regression']['hyperparameters'][idx]
model_logistic = LogisticRegression(**best_params_logistic, max_iter=1000)
model_logistic.fit(data_preprocess.train, data_preprocess.label_train)
predictions_logistic = model_logistic.predict(data_preprocess.test)

# Save Logistic Regression Predictions
with open('output_labels_logistic_classifier_balanced.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])
    for i, label in enumerate(predictions_logistic):
        writer.writerow([i, label])

# Best XGBoost Model
idx = np.argmax(np.array(data['xgboost']['f1_scores']))
best_params_xgboost = data['xgboost']['hyperparameters'][idx]
model_xgboost = XGBClassifier(**best_params_xgboost)
model_xgboost.fit(data_preprocess.train, data_preprocess.label_train)
predictions_xgboost = model_xgboost.predict(data_preprocess.test)

# Save XGBoost Predictions
with open('output_labels_xgboost_classifier_balanced.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])
    for i, label in enumerate(predictions_xgboost):
        writer.writerow([i, label])

idx = np.argmax(np.array(data['bayes']['f1_scores']))
best_params_bayes = data['bayes']['hyperparameters'][idx]
model_bayes = MultinomialNB(**best_params_bayes)
model_bayes.fit(data_preprocess.train, data_preprocess.label_train)
predictions_bayes = model_bayes.predict(data_preprocess.test)

# Save XGBoost Predictions
with open('output_labels_bayes_classifier_balanced.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])
    for i, label in enumerate(predictions_bayes):
        writer.writerow([i, label])


idx = np.argmax(np.array(data['complement_bayes']['f1_scores']))
best_params_complement_bayes = data['complement_bayes']['hyperparameters'][idx]
model_complement_bayes = ComplementNB(**best_params_complement_bayes)
model_complement_bayes.fit(data_preprocess.train, data_preprocess.label_train)
predictions_complement_bayes = model_complement_bayes.predict(data_preprocess.test)

# Save XGBoost Predictions
with open('output_labels_complement_bayes_classifier_balanced.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])
    for i, label in enumerate(predictions_complement_bayes):
        writer.writerow([i, label])





