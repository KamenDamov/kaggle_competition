from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, make_scorer
import numpy as np
from sklearn.naive_bayes import ComplementNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score)

def create_pipeline_and_params(model_name):
    # TFIDF génèrent de moins bonnes perfo sur validation
    if model_name == 'ComplementNB':
        pipeline = Pipeline([
            #('tfidf', TfidfTransformer()),
            #('to_float32', FunctionTransformer(lambda X: csr_matrix(X, dtype=np.float32))),
            ('model', ComplementNB())
        ])
        param_grid = {'model__alpha': uniform(0.001, 0.7)}
    
    elif model_name == 'XGBoost':
        pipeline = Pipeline([
            #('tfidf', TfidfTransformer()),
            #('to_float32', FunctionTransformer(lambda X: csr_matrix(X, dtype=np.float32))),
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])
        param_grid = {
            'model__learning_rate': uniform(0.01, 0.2),         # Learning rate for the model
            'model__n_estimators': [200, 300, 400, 500],             # Number of boosting rounds
            'model__max_depth': [3, 5, 7, 10],                 # Maximum depth of each tree
            'model__subsample': uniform(0.6, 0.4)               # Fraction of samples used for each boosting round
        }
    
    elif model_name == 'LogisticRegression':
        pipeline = Pipeline([
            #('tfidf', TfidfTransformer()),
            #('to_float32', FunctionTransformer(lambda X: csr_matrix(X, dtype=np.float32))),
            ('model', LogisticRegression(max_iter=1000))
        ])
        param_grid = {'model__C': uniform(0.1, 10), 'model__penalty': ['l1'], 'model__solver': ['liblinear']}

    elif model_name == 'SVC':
        pipeline = Pipeline([
            #('tfidf', TfidfTransformer()),
            #('to_float32', FunctionTransformer(lambda X: csr_matrix(X, dtype=np.float32))),
            ('model', SVC(probability=True))
        ])
        param_grid = {
            'model__C': uniform(0.1, 10),
            'model__kernel': ['linear', 'rbf']
        }

    else:
        raise ValueError(f"Model {model_name} is not defined.")
    
    return pipeline, param_grid

def tune_model(pipeline, param_grid, X_train, y_train):
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, scoring=scorer, cv=cv,
        n_iter=15, n_jobs=1, random_state=0, verbose=3
    )
    random_search.fit(X_train, y_train)
    print(f"Best F1 for {pipeline.named_steps['model'].__class__.__name__}:", random_search.best_score_)
    y_pred = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, method='predict')
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        y_val_true = y_train[val_idx]
        y_val_pred = y_pred[val_idx]
        cm = confusion_matrix(y_val_true, y_val_pred)
        print(f'Confusion Matrix - Fold {fold + 1}:\n{cm}\n')
    
    return random_search.best_estimator_

def train_ensemble(X_train, y_train, model_names):

    tuned_models = []
    
    for model_name in model_names:
        pipeline, param_grid = create_pipeline_and_params(model_name)
        best_model = tune_model(pipeline, param_grid, X_train, y_train)
        tuned_models.append((model_name.lower(), best_model))
    
    ensemble = VotingClassifier(estimators=tuned_models, voting='soft')
    ensemble.fit(X_train, y_train)
    
    return ensemble

def estimate(X_test, model):
    return model.predict(X_test)