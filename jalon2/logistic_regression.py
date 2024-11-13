import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score)

def train_logreg(X_train, y_train): 
    """Entrainer regréssion logistique en lançant une validation croisée aléatoire"""
    pipeline = Pipeline([
            ('model', LogisticRegression(max_iter=1000))
        ])
    param_grid = {'model__C': uniform(0.1, 10), 'model__penalty': ['l1'], 'model__solver': ['liblinear']}

    random_search = RandomizedSearchCV(
        pipeline, param_grid, scoring=scorer, cv=5, n_jobs=1,
        verbose=3, n_iter=10, random_state=42
    )

    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)

    return random_search.best_params_, random_search.best_score_




    
