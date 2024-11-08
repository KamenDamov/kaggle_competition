import numpy as np
from preprocess_data import DataPreprocess, tree_based_dimensionality_reduction, smote_oversampling, \
    boostrap_oversampling, random_undersampling, remove_cum_sum
import visualize_data
from ensemble_learning import train_ensemble, estimate
from complement_naive_bayes import train_cnb_with_tfidf

from  save_output import save_output

if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    data_preprocess.remove_stopwords()
    indeces_to_remove = remove_cum_sum(data_preprocess.train, 0.95)
    data_preprocess.train = np.delete(data_preprocess.train, indeces_to_remove, axis=1)
    data_preprocess.test = np.delete(data_preprocess.test, indeces_to_remove, axis=1)
    X_train_undersampled, y_train_undersampled = random_undersampling(data_preprocess.train, data_preprocess.label_train)

    # Methode de oversampling désuètes
    #X_train_oversampled, y_train_oversampled = smote_oversampling(X_train_undersampled, y_train_undersampled)
    #X_train_oversampled, y_train_oversampled = boostrap_oversampling(X_train_undersampled, y_train_undersampled)
    
    # Lancer le VotingClassifier
    model_names = ['ComplementNB', 'XGBoost', 'SGD'] #, 'SVC'
    best_ensemble_model = train_ensemble(X_train_undersampled, y_train_undersampled, model_names)
    predictions_voter = best_ensemble_model.predict(data_preprocess.test)
    save_output(predictions_voter, "ensemble_cnb_xgboost_sgd", "random_search_10_iter", "stopwords_cum_sum_undersampled")
    
    # Lancer le Complement Naive Bayes
    best_cnb_model = train_cnb_with_tfidf(X_train_undersampled, y_train_undersampled)
    predictions_cnb = best_cnb_model.predict(data_preprocess.test)
    save_output(predictions_cnb, "cnb", "random_search_75_iter", "stopwords_undersampled")