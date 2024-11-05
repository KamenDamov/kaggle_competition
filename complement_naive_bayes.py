from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer

def train_cnb_with_tfidf(X_train, y_train) -> ComplementNB:
    # TFIDF génèrent de moins bonnes perfo sur validation
    pipeline = Pipeline([
        #('tfidf', TfidfTransformer()),   
        #('to_float32', FunctionTransformer(lambda X: X.astype(np.float32))),           
        ('cnb', ComplementNB())                    
    ])

    # Define parameter grid for both TF-IDF and ComplementNB
    param_distributions = {
        'cnb__alpha': uniform(0.0001, 1.5) 
    }

    scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        pipeline, param_distributions, scoring=scorer, cv=5, n_jobs=1, 
        verbose=3, n_iter=75, random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)
    best_cnb = random_search.best_estimator_
    return best_cnb
