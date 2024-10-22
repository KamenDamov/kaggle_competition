from preprocess_data import DataPreprocess
import csv
from xgboost_classifier import XGBoostClassifierPipeline

if __name__ == "__main__": 
    data_preprocess = DataPreprocess()
    print("data processed")
    data_preprocess.remove_stopwords()
    print("stopwords removed")
    data_preprocess.pca_train()

    xgboost = XGBoostClassifierPipeline()
    best_params = xgboost.hyperparameter_tuning(data_preprocess.train, data_preprocess.label_train)
    xgboost.fit_best(data_preprocess.train, data_preprocess.label_train, best_params)
    data_preprocess.pca_test()
    predictions = xgboost.best_model.predict(data_preprocess.test)
        

    with open('output_labels_xgboost_classifier.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["ID", "label"])

        for i, label in enumerate(predictions):
            writer.writerow([i, label])
