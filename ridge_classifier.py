import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier

from ensemble_learning import create_pipeline_and_params, tune_model
from preprocess_data import DataPreprocess, random_undersampling, remove_cum_sum
from save_output import save_output

if __name__ == '__main__':
    model_name = 'SGD'
    data_preprocess = DataPreprocess()

    indeces_to_remove = remove_cum_sum(data_preprocess.train, 0.95)
    data_preprocess.train = np.delete(data_preprocess.train, indeces_to_remove, axis=1)
    data_preprocess.test = np.delete(data_preprocess.test, indeces_to_remove, axis=1)

    X_train, y_train = random_undersampling(data_preprocess.train, data_preprocess.label_train)

    pipeline, param_grid = create_pipeline_and_params(model_name)
    best_model, best_params = tune_model(pipeline, param_grid, X_train, y_train)

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(data_preprocess.test)

    params = f"loss=log_loss-penalty=elasticnet-alpha={best_params['model__alpha']}-l1_ratio={best_params['model__l1_ratio']}"

    save_output(predictions, "ridge", params, "cum-sum_undersampling")

