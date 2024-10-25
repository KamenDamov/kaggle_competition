import csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# from cuml.model_selection import GridSearchCV
# from cuml.svm import SVC
from preprocess_data import DataPreprocess
from save_output import save_output


def grid_search(X, y):
    # TODO: need to change hyperparameters
    C_range = np.logspace(-1, 10, 5)[::-1]
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    print(param_grid)
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, scoring='f1', verbose=3)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


if __name__ == '__main__':
    data_preprocess = DataPreprocess()
    print('data processed')
    # data_preprocess.initialize_tfidf()
    # print('tfidf initialized!')
    # data_preprocess.apply_truncated_svd(100)
    # print('truncated svd applied!')

    train = data_preprocess.train
    label_train = data_preprocess.label_train
    test = data_preprocess.test

    # best_params_, best_score_ = grid_search(train, label_train)
    # print('best params: {}, score: {}'.format(best_params_, best_score_))

    svc = RandomForestClassifier()
    # svc = SVC(C=best_params_['C'], gamma=best_params_['gamma'])
    svc.fit(train, label_train)
    print('f1_score on train:', f1_score(label_train, svc.predict(train)))

    y_pred = svc.predict(test)

    params = f'C={svc.C}gamma={svc.gamma}'
    save_output(y_pred, "random_forests", params, "")