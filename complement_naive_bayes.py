from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import numpy as np

def train_cnb(X_train, y_train) -> ComplementNB:

    cnb = ComplementNB()

    # Define parameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0],  # Range of smoothing values
        'norm': [True, False]            # Whether to normalize weights
    }

    scorer = make_scorer(f1_score)  # 'weighted' accounts for class imbalance

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(cnb, param_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=2)

    # Fit the model with GridSearch
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print("Best Parameters:", best_params)
    print("Best F1 Score:", grid_search.best_score_)

    best_cnb = ComplementNB(**best_params)
    best_cnb.fit(X_train, y_train)
    return best_cnb

def estimate(X_test:np.array, model: ComplementNB) -> np.array:
    return model.predict(X_test)
