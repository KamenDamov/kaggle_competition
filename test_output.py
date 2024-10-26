import pandas as pd

df_1 = pd.read_csv("output/20241025/182936_comp_bayes_alpha=0.07900000000000001_stopwords.csv")
df_2 = pd.read_csv("submissions/0.73538_output_labels_bayes_classifier.csv")

print((df_1 != df_2)["label"].sum())