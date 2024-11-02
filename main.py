import numpy as np
from preprocess_data import DataPreprocess, tree_based_dimensionality_reduction, smote_oversampling
import visualize_data
from  save_output import save_output
from complement_naive_bayes import train_cnb, estimate

if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    data_preprocess.remove_non_words()
    X_train_reduced, indeces = tree_based_dimensionality_reduction(data_preprocess.train, data_preprocess.label_train)
    data_preprocess.vocab_map = data_preprocess.vocab_map[indeces]
    #X_train_reduced_oversampled, y_train_oversampled = smote_oversampling(X_train_reduced, data_preprocess.label_train)

    #visualize_data.get_graphs(data_preprocess.vocab_map, X_train_reduced_oversampled, y_train_oversampled)
    
    X_test = data_preprocess.test[:, indeces]
    best_model = train_cnb(X_train_reduced, data_preprocess.label_train)
    predictions = estimate(X_test, best_model)
    save_output(predictions, "cnb", "grid_search", "removed_nonwords_tree_reduction_smote")
    x = 3 