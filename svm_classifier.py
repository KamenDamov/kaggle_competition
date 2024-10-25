import csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
# from cuml.model_selection import GridSearchCV
# from cuml.svm import SVC
from preprocess_data import DataPreprocess
from save_output import save_output


def grid_search(X, y):
    C_range = [np.logspace(-2, 10, 13)[0]]
    gamma_range = [np.logspace(-9, 3, 13)[0]]
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

if __name__ == '__main__':
    data_preprocess = DataPreprocess()
    print('data processed')
    data_preprocess.initialize_tfidf()
    print('tfidf initialized!')
    data_preprocess.apply_truncated_svd(100)
    print('truncated svd applied!')

    best_params_, best_score_ = grid_search(data_preprocess.train_tfidf, data_preprocess.label_train)
    print('best params: {}, score: {}'.format(best_params_, best_score_))

    # svc = SVC()
    svc = SVC(C=0.01, gamma=0.01)
    svc.fit(data_preprocess.train_tfidf, data_preprocess.label_train)
    print('f1_score on train:', f1_score(data_preprocess.label_train, svc.predict(data_preprocess.train_tfidf)))

    y_pred = svc.predict(data_preprocess.test_tfidf)

    params = f'C={svc.C}gamma={svc.gamma}'
    save_output(y_pred, "svm", params, "")