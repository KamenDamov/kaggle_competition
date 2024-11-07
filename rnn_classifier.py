import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import numpy as np
from keras import backend
from preprocess_data import DataPreprocess
from save_output import save_output


def plot_accuracy(history):
    # Plot accuracy
    plt.plot(history.history['f1_score_keras'], label='Train f1')
    plt.plot(history.history['val_f1_score_keras'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('f1_score')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def grid_search(X, y, model):
    alpha = np.arange(0.001, 0.1, 0.001)
    param_grid = dict(alpha=alpha)
    print(param_grid)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='f1', verbose=3)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


def f1_score_keras(y_true, y_pred):
    y_pred = backend.round(y_pred)  # Round predictions to 0 or 1
    tp = backend.sum(backend.cast(y_true * y_pred, 'float'), axis=0)  # True positives
    fp = backend.sum(backend.cast((1 - y_true) * y_pred, 'float'), axis=0)  # False positives
    fn = backend.sum(backend.cast(y_true * (1 - y_pred), 'float'), axis=0)  # False negatives

    precision = tp / (tp + fp + backend.epsilon())
    recall = tp / (tp + fn + backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + backend.epsilon())
    return backend.mean(f1)

def random_split(X: np.array, y: np.array, split_ratio: float = 0.9):
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * split_ratio)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


# Define the Sparse Neural Network model
class SparseNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(SparseNeuralNetwork, self).__init__()
        # Define layers with small sizes to avoid overfitting due to high dimensionality
        self.dense1 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Convert sparse input to dense for compatibility with Keras Dense layers
        if isinstance(inputs, tf.SparseTensor):
            dense_inputs = tf.sparse.to_dense(inputs)
        else:
            dense_inputs = inputs  # Input is already dense

        # Apply dense layers with activations
        x = self.dense1(dense_inputs)
        x = self.dropout(x, training=True)  # Apply dropout during training

        x = self.dense2(x)
        x = self.dropout(x, training=True)

        # Final output layer
        return self.output_layer(x)


def main():
    data_preprocess = DataPreprocess()
    print("data processed")
    data_preprocess.remove_stopwords()
    print("stopwords removed")
    # data_preprocess.initialize_tfidf()
    # print("tf-idf processed")
    # data_preprocess.apply_truncated_svd(100)
    # print('truncated svd applied!')
    # data_preprocess.remove_min_max(10, 0.999)
    # print("min-max removed")

    train = data_preprocess.train
    label_train = data_preprocess.label_train
    test = data_preprocess.test
    print('train:', train.shape)
    print('test:', test.shape)
    print('Ratio of 1 in train:', np.sum(label_train == 1) / len(label_train))

    X_train, y_train, X_val, y_val = random_split(train, label_train, 0.9)

    # Define the model
    # model = Sequential([
    #     Dense(128, activation='relu', input_shape=(train.shape[1],)),  # Input layer with 128 units
    #     Dense(64, activation='relu'),  # Hidden layer with 64 units
    #     Dense(1, activation='sigmoid')  # Output layer with 1 unit for binary output
    # ])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score_keras])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Instantiate and compile the model
    model = SparseNeuralNetwork()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1_score_keras])

    nonzero_indices = np.nonzero(X_train)
    print(np.array([nonzero_indices[0].shape, nonzero_indices[1].shape]).T)
    print(X_train[nonzero_indices].shape)
    X_train_tensor = tf.sparse.SparseTensor(
        indices=np.array([nonzero_indices[0], nonzero_indices[1]]).T,
        values=X_train[nonzero_indices],
        dense_shape=X_train.shape
    )

    # Train the model
    # model.fit(X_train_tensor, y_train, epochs=10, batch_size=32)

    # best_params_, best_score_ = grid_search(train, label_train, model)
    # print('best params: {}, score: {}'.format(best_params_, best_score_))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_tensor, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    train_loss, train_f1 = model.evaluate(X_train, y_train)
    print("final loss:", train_loss, "final f1_score:", train_f1)

    print(f"f1_score on train: {f1_score(label_train, (np.array([x[0] for x in model.predict(train)]) > 0.5).astype(int))}")
    plot_accuracy(history)

    # Assuming `new_test_array` is a new bag-of-words matrix for prediction
    predictions = model.predict(test)
    y_pred = (np.array([x[0] for x in predictions]) > 0.5).astype(int)  # Convert probabilities to binary labels (0 or 1)

    params = f'epochs={10}batch_size={32}'
    save_output(y_pred, "rnn", params, "stopwords_tf-idf_min-max")


if __name__ == "__main__":
    main()

