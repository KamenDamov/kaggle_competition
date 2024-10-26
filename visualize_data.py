import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from preprocess_data import DataPreprocess


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()

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
    top_idx = np.argsort(data)[-10:][::-1]
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
    data_1 = np.sort(dataset_1.sum(axis=0))
    data_0 = np.sort(dataset_0.sum(axis=0))
    data = np.sort(dataset.sum(axis=0))
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


def visualize_2d_svd(dataset_1, dataset_0):
    # Unzip points into x and y coordinates for each label group
    x_0, y_0 = zip(*dataset_0)
    x_1, y_1 = zip(*dataset_1)

    # Create scatter plot
    plt.scatter(x_0, y_0, color='blue', label='Label 0')
    plt.scatter(x_1, y_1, color='red', label='Label 1')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Points Scatter Plot with Labels')

    # Show legend
    plt.legend()

    # Display plot
    plt.show()

def visualize_3d_svd(dataset_1, dataset_0):
    # Unzip points into x, y, and z coordinates for each label group
    x_0, y_0, z_0 = zip(*dataset_0)
    x_1, y_1, z_1 = zip(*dataset_1)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points with label 0
    ax.scatter(x_0, y_0, z_0, color='blue', label='Label 0')

    # Plot points with label 1
    ax.scatter(x_1, y_1, z_1, color='red', label='Label 1')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points Scatter Plot with Labels')

    # Show legend
    ax.legend()

    # Display plot
    plt.show()

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


# def main():
#     data_preprocess = DataPreprocess()
#     vocab_map = data_preprocess.vocab_map
#     for dataset, labels in [(data_preprocess.train, data_preprocess.label_train), (data_preprocess.train_tfidf, data_preprocess.label_train)]:
#         # Créer un dataset pour les labels 1 et les labels 0
#         dataset_1, dataset_0 = [], []
#         n = len(dataset) // 2
#         if labels is None:
#             dataset_1 = dataset
#         else:
#             for i in range(n):
#                 if int(labels[i]) == 1:
#                     dataset_1.append(dataset[i])
#                     dataset_1.append(dataset[i + n])
#                 else:
#                     dataset_0.append(dataset[i])
#                     dataset_0.append(dataset[i + n])
#         dataset_1, dataset_0 = np.array(dataset_1), np.array(dataset_0)
#
#     length_of_docs(dataset_1, dataset_0, dataset)
#     # length_of_words(dataset_1, dataset_0, dataset)
#     # mean_length_of_words(dataset_1, dataset_0, dataset)
#     most_frequent_words(dataset, vocab_map, 'all labels')
#     most_frequent_words(dataset_1, vocab_map, 'label 1')
#     most_frequent_words(dataset_0, vocab_map, 'label 0')
#     distribution_of_words(dataset_1, dataset_0, dataset)

def main():
    print("processing data...")
    data_preprocess = DataPreprocess()
    vocab_map = data_preprocess.vocab_map
    print("data processed!")

    print("applying tf-idf...")
    data_preprocess.initialize_tfidf()
    print("tf-idf applied!")

    data_preprocess.apply_truncated_svd(2)
    visualize_2d_svd(data_preprocess.train, data_preprocess.test)
    print("2d visualization completed!")

    print("processing data...")
    data_preprocess = DataPreprocess()
    vocab_map = data_preprocess.vocab_map
    print("data processed!")
    data_preprocess.apply_truncated_svd(3)
    visualize_3d_svd(data_preprocess.train, data_preprocess.test)
    print("3d visualization completed!")
    # get_graphs(vocab_map, data_preprocess.train, data_preprocess.label_train)
    # print("graphs generated!")
    # print("removing stopwords...")
    # data_preprocess.remove_stopwords()
    # print("stopwords removed!")
    # print("generating graphs...")
    # get_graphs(vocab_map, data_preprocess.train, data_preprocess.label_train)
    # print("graphs generated!")
    # print("generating graphs...")
    # get_graphs(vocab_map, data_preprocess.train_tfidf, data_preprocess.label_train)
    # print("graphs generated!")


if __name__ == "__main__":
    main()
