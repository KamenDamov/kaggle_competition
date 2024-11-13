from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer

def train_cnb_with_tfidf(X_train, y_train) -> ComplementNB:
    """Entraine un modèle de Complement Naive Bayes en appliquant une transformation de TF-IDF"""

    # Créer une pipeline qui applique les transformations avant le modèle
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),   
        ('to_float32', FunctionTransformer(lambda X: X.astype(np.float32))),           
        ('cnb', ComplementNB())                    
    ])

    # Définir la distribution des paramètres à évaluer
    param_distributions = {
        'cnb__alpha': uniform(0.0001, 1.5) 
    }

    # Valider à l'aide du score f1
    scorer = make_scorer(f1_score)

    # Faire 10 itérations avec 5-folds
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, scoring=scorer, cv=5, n_jobs=1,
        verbose=3, n_iter=10, random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)
    best_cnb = random_search.best_estimator_
    tfidf_transformer = best_cnb.named_steps['tfidf']
    return best_cnb, tfidf_transformer


def train_cnb(X_train, y_train) -> ComplementNB:
    """Entraine un modèle de Complement Naive Bayes"""

    pipeline = Pipeline([
        ('cnb', ComplementNB())                    
    ])

    # Définir la distribution des paramètres à évaluer
    param_distributions = {
        'cnb__alpha': uniform(0.0001, 1.5) 
    }

    # Valider à l'aide du score f1
    scorer = make_scorer(f1_score)

    # Faire 10 itérations avec 5-folds
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, scoring=scorer, cv=5, n_jobs=1,
        verbose=3, n_iter=10, random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters:", best_params)
    print("Best F1 Score:", random_search.best_score_)
    best_cnb = random_search.best_estimator_
    return best_cnb
