import pandas as pd

champion = pd.read_csv("output/20241104/134731_ensemble_cnb_xgb_logreg_random_search_stopwords_undersampled.csv")
runner_up = pd.read_csv("184703_comp_bayes_alpha0.761_stopwords.csv")
candidat = pd.read_csv("output/20241107/140736_cnb_random_search_50_iter_cumulative_sum_undersampled.csv")


print((champion != candidat)["label"].sum())