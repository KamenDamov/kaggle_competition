import numpy as np
from preprocess_data import DataPreprocess, tree_based_dimensionality_reduction, smote_oversampling, boostrap_oversampling, random_undersampling
import visualize_data
from ensemble_learning import train_ensemble, estimate
from  save_output import save_output

if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    data_preprocess.remove_stopwords()
    #visualize_data.get_graphs(data_preprocess.vocab_map, data_preprocess.train, data_preprocess.label_train)
    #X_train_reduced, indeces = tree_based_dimensionality_reduction(data_preprocess.train, data_preprocess.label_train)
    #data_preprocess.vocab_map = data_preprocess.vocab_map[indeces]
    X_train_undersampled, y_train_undersampled = random_undersampling(data_preprocess.train, data_preprocess.label_train)
    #visualize_data.get_graphs(data_preprocess.vocab_map, X_train_undersampled, y_train_undersampled)
    #X_train_oversampled, y_train_oversampled = smote_oversampling(data_preprocess.train, data_preprocess.label_train)

    X_train_oversampled, y_train_oversampled = boostrap_oversampling(X_train_undersampled, y_train_undersampled)
    #visualize_data.get_graphs(data_preprocess.vocab_map, X_train_reduced_oversampled, y_train_oversampled)
    
    model_names = ['ComplementNB', 'XGBoost', 'LogisticRegression', 'SVC']
    best_ensemble_model = train_ensemble(X_train_oversampled, y_train_oversampled, model_names)
    X_test_tfidf = best_ensemble_model.named_estimators_['complementnb'].named_steps['tfidf'].transform(data_preprocess.test)
    predictions = best_ensemble_model.predict(X_test_tfidf)
    save_output(predictions, "ensemble_cnb_xgb_logreg_svc", "random_search", "stopwords_tfidf_undersampled_smote")