import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from cuml.svm import SVC
from sklearn.metrics import f1_score
from preprocess_data import DataPreprocess
from save_output import save_output

def train_svc(X, y):
    """Entraine un modèle de SVM avec noyau RBF"""

    # Définir la distribution des paramètres à évaluer
    C_range = np.arange(45, 55, 1)
    gamma_range = np.arange(0.003, 0.02, 0.001)
    param_grid = dict(gamma=gamma_range, C=C_range)
    print(param_grid)

    # Faire 10 itérations avec 5-folds en validant sur f1
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv, scoring='f1', verbose=3)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


if __name__ == '__main__':
    data_preprocess = DataPreprocess()
    print('data processed')

    train = data_preprocess.train
    label_train = data_preprocess.label_train
    test = data_preprocess.test
    print('Ratio of 1 in train:', np.sum(label_train == 1) / len(label_train))

    best_params_, best_score_ = train_svc(train, label_train)
    print('best params: {}, score: {}'.format(best_params_, best_score_))

    svc = SVC(kernel='rbf', C=best_params_['C'], gamma=best_params_['gamma'])
    svc.fit(train, label_train)
    print('f1_score on train:', f1_score(label_train, svc.predict(train)))

    y_pred = svc.predict(test)

    params = f'kernel={svc.kernel}C={svc.C}gamma={svc.gamma}'
    save_output(y_pred, "svm", params, "")