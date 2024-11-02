import pandas as pd

df_1 = pd.read_csv("output_labels_complement_bayes_classifier_balanced.csv")
df_2 = pd.read_csv("output_labels_bayes_classifier.csv")

print((df_1 != df_2)["label"].sum())

print(df_1["label"].groupby(df_1["label"]).count())
print(df_2["label"].groupby(df_2["label"]).count())