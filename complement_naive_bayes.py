from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer

def train_cnb_with_tfidf(X_train, y_train) -> ComplementNB:
    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),   
        ('to_float32', FunctionTransformer(lambda X: X.astype(np.float32))),           # Then apply TF-IDF transformation
        ('cnb', ComplementNB())                     # Finally, Complement Naive Bayes classifier
    ])

    # Define parameter grid for both TF-IDF and ComplementNB
    param_distributions = {
        'cnb__alpha': uniform(0.1, 10)  # Uniform distribution from 0.1 to 2.1 for alpha
    }

    # Define the scorer using weighted F1 score to account for class imbalance
    scorer = make_scorer(f1_score, average='weighted')

    # Set up GridSearchCV with cross-validation on the pipeline
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, scoring=scorer, cv=5, n_jobs=-1, 
        verbose=3, n_iter=10, random_state=0  # Adjust n_iter for more/less trials
    )

    # Fit the model with GridSearchCV
    random_search.fit(X_train, y_train)

    # Get the best parameters and the best model
    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)

    # Best model is already fitted by GridSearchCV
    best_cnb = random_search.best_estimator_
    return best_cnb

def estimate(X_test, model) -> np.array:
    return model.predict(X_test)