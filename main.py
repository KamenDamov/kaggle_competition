import numpy as np
from preprocess_data import DataPreprocess, random_undersampling, get_indices_to_remove_cum_sum
import visualize_data
from ensemble_learning import train_ensemble, estimate
from complement_naive_bayes import train_cnb_with_tfidf

from  save_output import save_output

if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    #visualize_data.check_sparsity_decrease(data_preprocess.train)

    #indeces_to_remove = remove_cum_sum(np.concatenate((data_preprocess.train, data_preprocess.test), axis=0), 0.99)
    #indeces_to_remove = remove_cum_sum(data_preprocess.train, 0.95)
    #data_preprocess.train = np.delete(data_preprocess.train, indeces_to_remove, axis=1)
    #data_preprocess.test = np.delete(data_preprocess.test, indeces_to_remove, axis=1)

    X_train_undersampled, y_train_undersampled = random_undersampling(data_preprocess.train, data_preprocess.label_train)

    # Lancer le VotingClassifier
    model_names = ['ComplementNB', 'XGBoost', 'LogisticRegression'] #, 'SVC' 
    best_ensemble_model = train_ensemble(X_train_undersampled, y_train_undersampled, model_names)
    predictions_voter = best_ensemble_model.predict(data_preprocess.test)
    save_output(predictions_voter, "ensemble_cnb_xgboost_logreg", "random_search_15_iter", "cumulative_sum_undersampled")
    
    # Lancer le Complement Naive Bayes
    #best_cnb_model = train_cnb_with_tfidf(X_train_undersampled, y_train_undersampled)
    #predictions_cnb = best_cnb_model.predict(data_preprocess.test)
    #save_output(predictions_cnb, "cnb", "random_search_50_iter", "cumulative_sum_undersampled")