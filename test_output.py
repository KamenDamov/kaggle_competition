import pandas as pd

df_1 = pd.read_csv("output/20241102/144312_cnb_grid_search_removed_nonwords_tree_reduction_smote.csv")
df_2 = pd.read_csv("output/20241102/143254_cnb_grid_search_removed_nonwords_tree_reduction_smote.csv")

print((df_1 != df_2)["label"].sum())