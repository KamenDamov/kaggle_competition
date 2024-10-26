import csv
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
# from cuml.model_selection import GridSearchCV
# from cuml.svm import SVC
from preprocess_data import DataPreprocess
from save_output import save_output


def grid_search(X, y):
    alpha = np.arange(50, 200, 10)
    param_grid = dict(alpha=alpha)
    print(param_grid)
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    grid = GridSearchCV(RidgeClassifier(), param_grid=param_grid, cv=cv, scoring='f1', verbose=3)
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
    print('train:', train.shape)
    print('test:', test.shape)
    print('Ratio of 1 in train:', np.sum(label_train == 1) / len(label_train))

    best_params_, best_score_ = grid_search(train, label_train)
    print('best params: {}, score: {}'.format(best_params_, best_score_))

    classifier = RidgeClassifier(alpha=best_params_['alpha'], solver="sparse_cg")
    classifier.fit(train, label_train)
    print('f1_score on train:', f1_score(label_train, classifier.predict(train)))

    y_pred = classifier.predict(test)

    params = f'alpha={classifier.alpha}'
    save_output(y_pred, "ridge", params, "")