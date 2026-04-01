import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from IPython.display import clear_output

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

def plot_loss(history):
    
    plt.figure()
    plt.title('Training performance (Loss)')
    plt.plot(history.epoch, history.history['loss'], label='train_loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy(history):

    plt.figure()
    plt.title('Accuracy performance')
    plt.plot(history.epoch, history.history['accuracy'], label='train_accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=None):
    if labels is None:
        labels = []
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=12)
    cbar = plt.colorbar(shrink=0.75, aspect=20, pad=0.02)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices_by_snr(model, X_test, Y_test, snr_test, snr_values, class_labels):
    accuracies = []
    confusion_matrices = {}

    for snr in snr_values:
        indices = np.where(snr_test == snr)[0]
        X_subset = X_test[indices]
        Y_subset = Y_test[indices]

        Y_pred = model.predict(X_subset)
        conf = np.zeros((len(class_labels), len(class_labels)))
        conf_norm = np.zeros((len(class_labels), len(class_labels)))

        for i in range(X_subset.shape[0]):
            true_class = np.argmax(Y_subset[i])
            pred_class = np.argmax(Y_pred[i])
            conf[true_class, pred_class] += 1

        for i in range(len(class_labels)):
            row_sum = np.sum(conf[i])
            if row_sum == 0:
                conf_norm[i] = 0
            else:
                conf_norm[i] = conf[i] / row_sum

        confusion_matrices[snr] = conf_norm

        plot_confusion_matrix(conf_norm, labels=class_labels, title=f"Confusion Matrix (SNR={snr})")

        correct = np.sum(np.diag(conf))
        total = np.sum(conf)
        accuracy = correct / total if total > 0 else 0
        print(f"Overall Accuracy {snr}: ", accuracy)
        accuracies.append(accuracy)

    return accuracies

def evaluate_model_by_snr(model_path, class_labels, snr_test, snr_values, data_gen, X_test, Y_test, custom_objects=None):
    model = load_model(model_path, custom_objects=custom_objects)
    score = model.evaluate(data_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    results = plot_confusion_matrices_by_snr(
        model, X_test, Y_test, snr_test, snr_values, class_labels
    )

    return results

def plot_accuracy_vs_snr(snr_values, accuracy_values, label='Model'):
    plt.figure(figsize=(11, 7))
    plt.plot(snr_values, accuracy_values, marker='s', label=label)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Accuracy vs SNRs")
    plt.show()