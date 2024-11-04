import pandas as pd

champion = pd.read_csv("184703_comp_bayes_alpha0.761_stopwords.csv")
runner_up = pd.read_csv("output\output_labels_bayes_classifier.csv")
candidat = pd.read_csv("output/20241104/134731_ensemble_cnb_xgb_logreg_random_search_stopwords_undersampled.csv")

print((champion != candidat)["label"].sum())