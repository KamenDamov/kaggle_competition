import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter

from preprocess_data import DataPreprocess, random_undersampling


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

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
def length_of_docs(dataset_1, dataset_0, dataset, title):
  data_1 = dataset_1.sum(axis=1)
  data_0 = dataset_0.sum(axis=1)
  data = dataset.sum(axis=1)
  plt.hist(data_1, color="#00FF0040")  # range pour éviter les valeurs aberrantes
  plt.hist(data_0, color="#FF000040")
  plt.hist(data, color="#0000FF40")
  plt.title(title)
  plt.xlabel("Longueur")
  plt.ylabel("Fréquence")
  plt.grid(False)
  plt.legend(["Label 0", "Label 1", "Tous docs"])
  plt.show()

def length_of_words(dataset_1, dataset_0, dataset):
    """Graphique pour la longueur des mots"""
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

def mean_length_of_words(dataset_similaire, dataset_assimilaire, dataset):
    """Longueur moyenne des mots par phrase"""
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

def most_frequent_words(dataset, vocab_map, title):
    """Top 10 mots les plus fréquents"""
    data = dataset.sum(axis=0)
    top_idx = np.argsort(data)[-10:][::-1]
    x, y= [], []
    for idx in top_idx:
        x.append(vocab_map[idx])
        y.append(data[idx])
    sns.barplot(x=y,y=x)
    plt.title(title)
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
    plt.yscale('log')
    plt.title("Distribution des mots")
    plt.xlabel("Mots")
    plt.ylabel("Fréquence relative")
    plt.grid(False)
    plt.legend(["Label 0", "Label 1", "Tous docs"])
    plt.show()

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

# 1. Word Frequency Histogram
def plot_word_frequency(train, vocab_map):
    word_counts = np.sum(train, axis=0)
    sorted_indices = np.argsort(word_counts)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar([vocab_map[i] for i in sorted_indices[:30]], word_counts[sorted_indices[:30]], color='skyblue')
    plt.xticks(rotation=90)
    plt.title("Top 30 Word Frequencies")
    plt.show()



# 3. Heatmap of the Bag-of-Words Matrix
def plot_bow_heatmap(train):
    plt.figure(figsize=(10, 8))
    sns.heatmap(train, cmap="YlGnBu", cbar=True)  # Displaying a subset for clarity
    plt.title("Bag-of-Words Matrix Heatmap (subset)")
    plt.xlabel("Words")
    plt.ylabel("Documents")
    plt.show()

# 4. TF-IDF Heatmap
from sklearn.feature_extraction.text import TfidfTransformer
def plot_tfidf_heatmap(train):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(train).toarray()
    plt.figure(figsize=(10, 8))
    sns.heatmap(tfidf, cmap="YlGnBu", cbar=True)
    plt.title("TF-IDF Matrix Heatmap (subset)")
    plt.xlabel("Words")
    plt.ylabel("Documents")
    plt.show()

# 5. Document Similarity Matrix
def plot_document_similarity(train):
    similarity_matrix = cosine_similarity(train)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="coolwarm")
    plt.title("Document Similarity Matrix")
    plt.show()

# 6. Dimensionality Reduction Scatter Plot (PCA & t-SNE)
def plot_dim_reduction(train, label_train, method='pca'):
    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, random_state=42)
    reduced_data = reducer.fit_transform(train)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=label_train, cmap="coolwarm", alpha=0.7)
    plt.title(f"2D {method.upper()} of Documents")
    plt.colorbar()
    plt.show()

# 7. Co-occurrence Network Graph
def plot_cooccurrence_network(train, vocab_map, threshold=0.2):
    cooccurrence = np.dot(train.T, train)
    G = nx.Graph()
    for i in range(len(vocab_map)):
        for j in range(i + 1, len(vocab_map)):
            if cooccurrence[i, j] > threshold:
                G.add_edge(vocab_map[i], vocab_map[j], weight=cooccurrence[i, j])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, edge_color="skyblue", font_size=10, alpha=0.7)
    plt.title("Word Co-occurrence Network")
    plt.show()

# 8. Class-Word Relationship Visualization
def plot_class_word_distribution(train, label_train, vocab_map):
    word_counts_class0 = np.sum(train[label_train == 0], axis=0)
    word_counts_class1 = np.sum(train[label_train == 1], axis=0)
    sorted_indices = np.argsort(word_counts_class1 - word_counts_class0)[::-1]
    plt.figure(figsize=(12, 10))
    plt.bar([vocab_map[i] for i in sorted_indices[:30]], (word_counts_class1 - word_counts_class0)[sorted_indices[:30]], color='salmon')
    plt.xticks(rotation=90)
    plt.title("Top 30 Class-Word Differences")
    plt.show()

def plot_full_word_frequency_distribution(train, label_train, title):
    train_0 = train[label_train==0]
    train_1 = train[label_train==1]
    word_counts = np.sum(train, axis=0)
    word_counts_0 = np.sum(train_0, axis=0)
    word_counts_1 = np.sum(train_1, axis=0)
    plt.figure(figsize=(14, 10))
    plt.hist(word_counts, bins=50, color='green')
    plt.hist(word_counts_0, bins=50, color='blue')
    plt.hist(word_counts_1, bins=50, color='red')
    plt.yscale('log')  # Use log scale for y-axis to handle wide range of frequencies
    plt.xlabel("Word Frequency")
    plt.ylabel("Number of Words")
    plt.title(title)
    plt.legend(["Tous docs", "Label 0", "Label 1"])
    plt.show()

def plot_cumulative_word_distribution(train, label_train, title):
    train_0 = train[label_train == 0]
    train_1 = train[label_train == 1]
    word_counts = np.sum(train, axis=0)
    word_counts_0 = np.sum(train_0, axis=0)
    word_counts_1 = np.sum(train_1, axis=0)
    sorted_counts = np.sort(word_counts)[::-1]
    sorted_counts_0 = np.sort(word_counts_0)[::-1]
    sorted_counts_1 = np.sort(word_counts_1)[::-1]
    cumulative_counts = np.cumsum(sorted_counts) / np.sum(sorted_counts)
    cumulative_counts_0 = np.cumsum(sorted_counts_0) / np.sum(sorted_counts_0)
    cumulative_counts_1 = np.cumsum(sorted_counts_1) / np.sum(sorted_counts_1)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_counts, color='green')
    plt.plot(cumulative_counts_0, color='blue')
    plt.plot(cumulative_counts_1, color='red')
    plt.xlabel("Words (sorted by frequency)")
    plt.ylabel("Cumulative Proportion of Total Word Count")
    plt.title(title)
    plt.grid(True)
    plt.legend(["Tous docs", "Label 0", "Label 1"])
    plt.show()


def plot_zipfs_law(train, label_train, title):
    train_0 = train[label_train == 0]
    train_1 = train[label_train == 1]
    word_counts = np.sum(train, axis=0)
    word_counts_0 = np.sum(train_0, axis=0)
    word_counts_1 = np.sum(train_1, axis=0)
    sorted_counts = np.sort(word_counts)[::-1]
    sorted_counts_0 = np.sort(word_counts_0)[::-1]
    sorted_counts_1 = np.sort(word_counts_1)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)
    ranks_0 = np.arange(1, len(sorted_counts_0) + 1)
    ranks_1 = np.arange(1, len(sorted_counts_1) + 1)

    plt.figure(figsize=(10, 10))
    plt.loglog(ranks, sorted_counts, color='green')
    plt.loglog(ranks_0, sorted_counts_0, color='blue')
    plt.loglog(ranks_1, sorted_counts_1, color='red')
    plt.xlabel("Rank of Word (log scale)")
    plt.ylabel("Frequency of Word (log scale)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(["Tous docs", "Label 0", "Label 1"])
    plt.show()

def get_graphs(vocab_map, dataset, labels, transformation):
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

    length_of_docs(dataset_1, dataset_0, dataset, "Documents frequency with " + transformation)
    most_frequent_words(dataset, vocab_map, "Top 10 most frequent words with " + transformation)
    plot_cumulative_word_distribution(dataset, labels, "Cumulative Word Frequency Distribution with " + transformation)
    plot_full_word_frequency_distribution(dataset, labels, "Overall Word Frequency Distribution with " + transformation)

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

    print("processing new data...")
    data_preprocess = DataPreprocess()
    print("new data processed!")
    print("removing by cumulative sum...")
    data_preprocess.remove_cum_sum()
    vocab_map = data_preprocess.vocab_map
    print("removed by cumulative sum!")
    print("generating graphs...")
    get_graphs(vocab_map, data_preprocess.train, data_preprocess.label_train)
    print("graphs generated!")

    print("processing new data...")
    data_preprocess = DataPreprocess()
    print("new data processed!")
    print("removing by undersampled...")
    X_train_undersampled, y_train_undersampled = random_undersampling(data_preprocess.train, data_preprocess.label_train)
    vocab_map = data_preprocess.vocab_map
    print("removed by undersampled!")
    print("generating graphs...")
    get_graphs(vocab_map, X_train_undersampled, y_train_undersampled)
    print("graphs generated!")

def main_2():
    data_preprocess = DataPreprocess()

    train = data_preprocess.train
    vocab_map = data_preprocess.vocab_map
    label_train = data_preprocess.label_train

    with open('english_stopwords', 'r') as f:
        stopwords = [line.strip() for line in f.readlines()]
    idx_stopwords = [i for i in range(len(vocab_map)) if vocab_map[i] in stopwords]
    plot_full_word_frequency_distribution(train[:,idx_stopwords], label_train, "Overall Stopwords Frequency Distribution")

    data_preprocess.apply_stemming()
    data_preprocess.apply_lemmatization()
    data_preprocess.remove_stopwords()
    train = data_preprocess.train
    vocab_map = data_preprocess.vocab_map
    label_train = data_preprocess.label_train

    with open('english_stopwords', 'r') as f:
        stopwords = [line.strip() for line in f.readlines()]
    idx_stopwords = [i for i in range(len(vocab_map)) if vocab_map[i] in stopwords]
    plot_full_word_frequency_distribution(train[:,idx_stopwords], label_train, "Overall Stopwords Frequency Distribution")

    print("Generating Word Frequency Histogram...")
    plot_word_frequency(train, vocab_map)

    print("Generating Heatmap of the Bag-of-Words Matrix...")
    # plot_bow_heatmap(train)

    print("Generating TF-IDF Heatmap...")
    # plot_tfidf_heatmap(train)

    print("Generating Document Similarity Matrix...")
    # plot_document_similarity(train)

    print("Generating 2D PCA Scatter Plot...")
    # plot_dim_reduction(train, label_train, method='pca')

    print("Generating 2D t-SNE Scatter Plot...")
    # plot_dim_reduction(train, label_train, method='tsne')

    print("Generating Co-occurrence Network Graph...")
    # plot_cooccurrence_network(train, vocab_map, threshold=0.2)

    print("Generating Class-Word Relationship Visualization...")
    plot_class_word_distribution(train, label_train, vocab_map)

    plot_full_word_frequency_distribution(train, label_train, "Overall Word Frequency Distribution")

    plot_cumulative_word_distribution(train, label_train, "Cumulative Word Frequency Distribution")

    plot_zipfs_law(train, label_train, "Zipf’s Law Plot")

    print("All plots generated.")


if __name__ == "__main__":
    main()
