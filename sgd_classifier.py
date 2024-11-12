import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score)

def train_sgd(X_train, y_train):
    pipeline = Pipeline([
        ('model', SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=10000))
    ])
    param_grid = {
        'model__alpha': np.logspace(-2, 0, 10),
        'model__l1_ratio': np.linspace(0.001, 1, 10)
    }

    random_search = random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, scoring=scorer, cv=cv,
        n_iter=10, n_jobs=1, random_state=0, verbose=3
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)
    best_sgd = random_search.best_estimator_
    return best_sgd
