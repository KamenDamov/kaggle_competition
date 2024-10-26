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
from visualize_data import plot_grid_search


def grid_search(X, y):
    C_range = np.arange(45, 55, 1)
    gamma_range = np.arange(0.003, 0.02, 0.001)
    param_grid = dict(gamma=gamma_range, C=C_range)
    print(param_grid)
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv, scoring='f1', verbose=3)
    grid.fit(X, y)
    plot_grid_search(grid.cv_results_, C_range, gamma_range, 'C', 'gamma')
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
    print('train:', train.shape)
    print('test:', test.shape)
    print('Ratio of 1 in train:', np.sum(label_train == 1) / len(label_train))

    best_params_, best_score_ = grid_search(train, label_train)
    print('best params: {}, score: {}'.format(best_params_, best_score_))

    # svc = SVC(kernel='linear')
    svc = SVC(kernel='linear', C=best_params_['C'], gamma=best_params_['gamma'])
    svc.fit(train, label_train)
    print('f1_score on train:', f1_score(label_train, svc.predict(train)))

    y_pred = svc.predict(test)

    params = f'kernel={svc.kernel}C={svc.C}gamma={svc.gamma}'
    save_output(y_pred, "svm", params, "")