import csv
import numpy as np

from preprocess_data import DataPreprocess


class SVM:
    def __init__(self, C=1.0, gamma=0.5, n_iters=1000):
        self.C = C  # Regularization parameter
        self.gamma = gamma  # Kernel parameter
        self.n_iters = n_iters
        self.alpha = None  # Lagrange multipliers
        self.b = 0  # Bias
        self.X_train = None
        self.y_train = None

    # RBF Kernel function
    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    # Fit the SVM model using the RBF kernel
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = np.where(y <= 0, -1, 1)  # Convert to {-1, 1}

        # Initialize alpha values (Lagrange multipliers)
        self.alpha = np.zeros(n_samples)

        # Train using gradient ascent on the dual problem
        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = 1 - self.y_train[i] * self._decision_function(X[i])
                if condition > 0:
                    self.alpha[i] += self.C * condition
                    self.b += self.C * self.y_train[i]

    # Decision function using the RBF kernel
    def _decision_function(self, X):
        result = 0
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y_train[i] * self.rbf_kernel(self.X_train[i], X)
        return result - self.b

    # Predict labels for new data
    def predict(self, X):
        y_pred = []
        for sample in X:
            prediction = np.sign(self._decision_function(sample))
            y_pred.append(prediction)
        return np.array(y_pred)

if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    bayes_classifier = SVM()
    # best_lps, best_f1 = bayes_classifier.hyperparameter_tuning(data_preprocess.train, data_preprocess.label_train)
    # print(best_lps, best_f1)

    tuned_bayes_classifier = SVM()
    tuned_bayes_classifier.fit(data_preprocess.train, data_preprocess.label_train)
    predictions = np.array([tuned_bayes_classifier.predict(x_i) for x_i in data_preprocess.test])

    with open('output_labels_bayes_classifier_part_3.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["ID", "label"])

        for i, label in enumerate(predictions):
            writer.writerow([i, label])
