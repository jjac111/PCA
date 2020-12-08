from config import *
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from time import time
from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import Callback


def read_data(dir):
    images = []
    labels = []
    for filename in os.listdir(dir):
        img = Image.open(data_dir + filename).convert('L')
        img = np.asarray(img)
        images.append(img)

        l = filename.split('.')[1][:2]
        labels.append(l)

    images = np.array(images)
    labels = np.array(labels)
    mean = np.mean(images, axis=0).astype(np.uint8)
    print('Average face:')
    display(Image.fromarray(mean))

    scaler = StandardScaler()

    shape = images.shape
    images = np.array([im.flatten() for im in images])
    images = scaler.fit_transform(images)

    uniques = np.unique(labels)
    label_matrix = np.zeros((len(images), len(uniques)))
    for row, l in zip(label_matrix[:], labels):
        row[np.where(uniques == l)] = 1
    labels = label_matrix

    return images, shape, labels, uniques


def do_PCA(manually_select):
    images, shape, labels, uniques = read_data(data_dir)

    if manually_select:
        n_components = chosen_components
        pca = PCA(n_components, whiten=True).fit(images)
    else:
        pca = PCA(target_variance, whiten=True).fit(images)
        n_components = len(pca.explained_variance_)

    eigenvalues = pca.explained_variance_

    plt.plot(eigenvalues)
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue (Autovalor)')
    plt.title('Scree Plot')
    plt.show()

    # Show accumulated variance
    var_ratio = pca.explained_variance_ratio_[:n_components]
    plt.bar(np.arange(len(var_ratio)), var_ratio)
    plt.xlabel('Component')
    plt.ylabel('Variance ratio')
    plt.title('Captured variance ratio for each component')
    plt.show()

    print(f'The {n_components} components accumulate {round(sum(var_ratio) * 100, 2)}% of the Variance')

    eigenvectors = pca.components_

    for i, v in enumerate(eigenvectors[:10]):
        v = v - min(v)
        v = v / max(v)
        v = v * 255
        img = v.reshape(shape[1:])

        print(f'Eigenface {i}:')
        display(Image.fromarray(img).convert('L'))

    transformed = pca.transform(images)

    print('Transformed training samples:')
    display(pd.DataFrame(transformed))

    return pca, transformed, labels, uniques


def do_training(data, labels, split=0.2):
    input_len = data.shape[1]
    inp = Input((input_len,))
    x = Dense(2 * input_len, activation='tanh')(inp)
    x = Dropout(dropout)(x)
    x = Dense(int(input_len / 2), activation='tanh')(x)
    x = Dropout(dropout)(x)
    out = Dense(labels.shape[1], activation='softmax')(x)

    model = Model(inp, out)

    optimizer = Adam(lr)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=split)

    for train_idx, test_idx in splitter.split(data, labels):
        pass

    train_X = np.take(data, train_idx, axis=0)
    train_y = np.take(labels, train_idx, axis=0)
    test_X = np.take(data, test_idx, axis=0)
    test_y = np.take(labels, test_idx, axis=0)

    start = time()
    history = model.fit(train_X, train_y,
                        validation_data=(test_X, test_y),
                        batch_size=batch_size, epochs=epochs,
                        verbose=0, callbacks=[PlotLearning()])
    # plt.plot(history.history['loss'], 'b', label='Loss')
    # plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    now = time() - start
    print(f'Finished training in {timedelta(seconds=round(now))}')
    model.save('model.h5')

    metrics = model.evaluate(test_X, test_y)

    print(f'Evaluation metrics:\n{model.metrics_names}\n{metrics}')


def do_testing(pca):
    images, shape, labels, uniques = read_data(test_dir)

    transformed = pca.transform(images)

    model = load_model('model.h5', compile=True)

    metrics = model.evaluate(transformed, labels)

    print(f'Evaluation metrics for prediction of new test samples:\n{model.metrics_names}\n{metrics}')



# Code by user kav, obtained from https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
# Used for live plotting the loss
class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure(figsize=(12, 5))

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        f.set_size_inches((12, 5))

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="validation loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        plt.show();
#############################################################
