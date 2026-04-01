import h5py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

class RadioMLConfig:
    CLASSES = ['OOK', '4ASK', '8ASK',  'BPSK', 'QPSK', '8PSK',
               '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
               '16QAM', '32QAM', '64QAM', '128QAM', '256QAM','AM-SSB-WC', 
               'AM-SSB-SC', 'AM-DSB-WC','AM-DSB-SC','FM', 'GMSK','OQPSK']
    FS = None
    N_SAMPLES = 1024
    SNR_RANGE = np.arange(-20, 31, 2)

def load_dataset(path, filename):
    file_path = os.path.join(path, filename)
    print(f"[INFO] Loading {file_path} ...")

    with h5py.File(file_path, "r") as f:
        X = f["X"][:]
        Y = f["Y"][:]
        Z = f["Z"][:]

    print(f"[INFO] Loaded Dataset | X:{X.shape} Y:{Y.shape} Z:{Z.shape} | dtype X:{X.dtype}")
    return X, Y, Z

def get_random_sample(X, Y, Z):
    idx = np.random.randint(0, X.shape[0])

    sample = X[idx]
    label = int(np.argmax(Y[idx]))
    snr_db = float(np.squeeze(Z[idx]))

    return {
        "index": idx,
        "signal": sample,
        "label": label,
        "snr_db": snr_db,
    }

def normalize_rms(iq_samples, eps=1e-10):
    
    i_comp = iq_samples[..., 0]
    q_comp = iq_samples[..., 1]
    power = i_comp**2 + q_comp**2
    rms = np.sqrt(np.mean(power, axis=1, keepdims=True) + eps)
    rms = np.expand_dims(rms, axis=-1)

    return iq_samples / rms

def split_dataset(X, Y, Z, val_size=0.15, test_size=0.15, random_state=2980):
    class_labels = np.argmax(Y, axis=1)

    temp_size = val_size + test_size
    X_train, X_temp, Y_train, Y_temp, Z_train, Z_temp = train_test_split(
        X, Y, Z,
        test_size=temp_size,
        random_state=random_state,
        stratify=class_labels
    )
    
    relative_test_size = test_size / temp_size 
    temp_class_labels = np.argmax(Y_temp, axis=1)
    X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(
        X_temp, Y_temp, Z_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_class_labels
    )
    
    return (X_train, Y_train, Z_train,
            X_val, Y_val, Z_val,
            X_test, Y_test, Z_test)

def build_tf_dataset(X, Y, batch_size=32, shuffle=False):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

class PyDataset(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = self.x[batch_indexes]
        batch_y = self.y[batch_indexes]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)