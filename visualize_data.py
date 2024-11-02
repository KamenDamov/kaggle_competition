import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from preprocess_data import DataPreprocess


def print_stats(title, data):
  print(f"-----{title}-----")
  print(f"Mean: {np.mean(data)}")
  print(f"STD: {np.std(data)}")
  print(f"Median: {np.median(data)}")
  print(f"Range: {np.ptp(data)}")
  print(f"IQR: {scipy.stats.iqr(data)}")
  print(f"Kurtosis: {scipy.stats.kurtosis(data)}")


# Longueur par document
def length_of_docs(dataset_1, dataset_0, dataset):
  data_1 = dataset_1.sum(axis=1)
  data_0 = dataset_0.sum(axis=1)
  data = dataset.sum(axis=1)
  plt.hist(data_1, color="#00FF0040")  # range pour éviter les valeurs aberrantes
  plt.hist(data_0, color="#FF000040")
  plt.hist(data, color="#0000FF40")
  plt.title("Fréquence de la longueur des documents")
  plt.xlabel("Longueur")
  plt.ylabel("Fréquence")
  plt.grid(False)
  plt.legend(["Label 0", "Label 1", "Tous docs"])
  plt.show()
  # print_stats("label 1", data_1)
  # print_stats("label 0", data_0)
  # print_stats("global", data)

# Longueur par mots
def length_of_words(dataset_1, dataset_0, dataset):
    # TODO: doit convertir les mots avec vocab_map
    data_1 = dataset_1
    data_0 = dataset_0
    data = dataset
    m = data.max()
    data.hist(range=(0, m), color="#00FF0040")  # range pour éviter les valeurs aberrantes
    data_1.hist(range=(0, m), color="#FF000040")
    data_0.hist(range=(0, m), color="#0000FF40")
    plt.title("Fréquence de la longueur des mots")
    plt.xlabel("Longueur")
    plt.ylabel("Fréquence")
    plt.grid(False)
    plt.legend(["Toutes phrases", "Phrases similaires", "Phrases non similaires"])
    plt.show()
    # print_stats("similaires", data_1)
    # print_stats("non similaires", data_0)
    # print_stats("global", data)

# Longueur moyenne des mots par phrase
def mean_length_of_words(dataset_similaire, dataset_assimilaire, dataset):
    # TODO: doit convertir les mots avec vocab_map
    data_sim = dataset_similaire.str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
    data_assim = dataset_assimilaire.str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
    data = dataset.str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
    data.hist(range=(0,15), color="#00FF0040")  # range pour éviter les valeurs aberrantes
    data_sim.hist(range=(0,15), color="#FF000040")
    data_assim.hist(range=(0,15), color="#0000FF40")
    plt.title("Fréquence de la longueur moyenne des mots par phrase")
    plt.xlabel("Longueur moyenne")
    plt.ylabel("Fréquence")
    plt.grid(False)
    plt.legend(["Toutes phrases", "Phrases similaires", "Phrases non similaires"])
    plt.show()
    # print_stats("similaires", data_sim)
    # print_stats("non similaires", data_assim)
    # print_stats("global", data)

# Top 10 mots les plus fréquents
def most_frequent_words(dataset, vocab_map, name):
    # TODO: mots les plus frequents par doc
    data = dataset.sum(axis=0)
    top_idx = np.argsort(data)[-20:][::-1]
    x, y= [], []
    for idx in top_idx:
        x.append(vocab_map[idx])
        y.append(data[idx])
    sns.barplot(x=y,y=x)
    plt.title(name + ": Top 10 mots les plus fréquents")
    plt.xlabel("Fréquence")
    plt.ylabel("Mot")
    plt.show()


def distribution_of_words(dataset_1, dataset_0, dataset):
    data_1 = np.sort(dataset_1.sum(axis=0) / dataset_1.sum())
    data_0 = np.sort(dataset_0.sum(axis=0) / dataset_0.sum())
    data = np.sort(dataset.sum(axis=0) / dataset.sum())
    idx = range(len(data))
    plt.plot(data_1)
    plt.plot(data_0)
    plt.plot(data)
    # plt.hist(data_1, bins=10, color="#00FF0040")  # range pour éviter les valeurs aberrantes
    # plt.hist(data_0, bins=10, color="#FF000040")
    # plt.hist(data, bins=10, color="#0000FF40")
    plt.yscale('log')
    plt.title("Distribution des mots")
    plt.xlabel("Mots")
    plt.ylabel("Fréquence relative")
    plt.grid(False)
    plt.legend(["Label 0", "Label 1", "Tous docs"])
    plt.show()
    # print_stats("label 1", data_1)
    # print_stats("label 0", data_0)
    # print_stats("global", data)


def get_graphs(vocab_map, dataset, labels):
    # Créer un dataset pour les labels 1 et les labels 0
    dataset_1, dataset_0 = [], []
    n = len(dataset) // 2
    if labels is None:
        dataset_1 = dataset
    else:
        for i in range(n):
            if int(labels[i]) == 1:
                dataset_1.append(dataset[i])
                dataset_1.append(dataset[i + n])
            else:
                dataset_0.append(dataset[i])
                dataset_0.append(dataset[i + n])
    dataset_1, dataset_0 = np.array(dataset_1), np.array(dataset_0)

    length_of_docs(dataset_1, dataset_0, dataset)
    # length_of_words(dataset_1, dataset_0, dataset)
    # mean_length_of_words(dataset_1, dataset_0, dataset)
    most_frequent_words(dataset, vocab_map, 'all labels')
    most_frequent_words(dataset_1, vocab_map, 'label 1')
    most_frequent_words(dataset_0, vocab_map, 'label 0')
    distribution_of_words(dataset_1, dataset_0, dataset)

def main():
    print("processing data...")
    data_preprocess = DataPreprocess()
    vocab_map = data_preprocess.vocab_map
    print("data processed!")
    get_graphs(vocab_map, data_preprocess.train, data_preprocess.label_train)
    print("graphs generated!")
    print("removing stopwords...")
    data_preprocess.remove_stopwords()
    print("stopwords removed!")
    print("generating graphs...")
    get_graphs(vocab_map, data_preprocess.train, data_preprocess.label_train)
    print("graphs generated!")
    print("applying tf-idf...")
    data_preprocess.initialize_tfidf()
    print("tf-idf applied!")
    print("generating graphs...")
    get_graphs(vocab_map, data_preprocess.train_tfidf, data_preprocess.label_train)
    print("graphs generated!")


if __name__ == "__main__":
    main()
