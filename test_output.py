import pandas as pd

df_1 = pd.read_csv("output\output_labels_bayes_classifier.csv")
df_2 = pd.read_csv("184703_comp_bayes_alpha0.761_stopwords.csv")

print((df_1 != df_2)["label"].sum())