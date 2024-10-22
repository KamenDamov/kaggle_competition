import numpy as np
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
optuna.logging.set_verbosity(optuna.logging.INFO)

class XGBoostClassifierPipeline:
    best_model: XGBClassifier
    def __init__(self):
        best_model = None

    def fit_best(self, X: np.array, y: np.array, params:dict):
        model = XGBClassifier(**params)
        model.fit(X, y)
        self.best_model = model

    def hyperparameter_tuning(self, X: np.array, y: np.array) -> dict:
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            model = XGBClassifier(**param)
            f1_scorer = make_scorer(f1_score, average='binary')
            scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        print("Best hyperparameters:", study.best_params)
        return study.best_params