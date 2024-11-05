import pandas as pd

df_1 = pd.read_csv("submissions/0.73543_184703_comp_bayes_alpha0.761_stopwords.csv")
df_2 = pd.read_csv("output/20241026/225613_comp_bayes_alpha=0.023_tf-idf_min-max.csv")
# df_2 = pd.read_csv("submissions/0.73543_184703_comp_bayes_alpha0.761_stopwords.csv")

print((df_1 != df_2)["label"].sum())