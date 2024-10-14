import pandas as pd

df_1 = pd.read_csv("output_labels_bayes_classifier_part_3.csv")
df_2 = pd.read_csv("output_labels_bayes_classifier.csv")

print((df_1 == df_2)["label"].sum())