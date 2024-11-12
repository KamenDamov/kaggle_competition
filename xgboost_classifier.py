import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

def train_xgboost_with_tfidf(X_train, y_train) -> XGBClassifier:
    # TFIDF génèrent de moins bonnes perfo sur validation
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),   
        ('to_float32', FunctionTransformer(lambda X: X.astype(np.float32))),           
        ('xgboost', XGBClassifier())                    
    ])

    # Define parameter grid for both TF-IDF and ComplementNB
    param_distributions = {
            'model__learning_rate': uniform(0.01, 0.2),         # Learning rate for the model
            'model__n_estimators': [200, 300, 400, 500],             # Number of boosting rounds
            'model__max_depth': [3, 5, 7, 10],                 # Maximum depth of each tree
            'model__subsample': uniform(0.6, 0.4)               # Fraction of samples used for each boosting round
        }

    scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        pipeline, param_distributions, scoring=scorer, cv=2, n_jobs=1, 
        verbose=3, n_iter=1, random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)
    best_xgboost = random_search.best_estimator_
    tfidf_transformer = best_xgboost.named_steps['tfidf']
    return best_xgboost, tfidf_transformer


def train_xgboost(X_train, y_train) -> XGBClassifier:
    pipeline = Pipeline([
        ('model', XGBClassifier())
    ])

    param_distributions = {
            'model__learning_rate': uniform(0.01, 0.2),         # Learning rate for the model
            'model__n_estimators': [200, 300, 400, 500],             # Number of boosting rounds
            'model__max_depth': [3, 5, 7, 10],                 # Maximum depth of each tree
            'model__subsample': uniform(0.6, 0.4)               # Fraction of samples used for each boosting round
        }

    scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        pipeline, param_distributions, scoring=scorer, cv=2, n_jobs=1, 
        verbose=3, n_iter=15, random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)
    best_xgboost = random_search.best_estimator_
    return best_xgboost


