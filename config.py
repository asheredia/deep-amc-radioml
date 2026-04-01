import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=2026):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs available:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not activate memory growth:", e)

def setup_environment(seed=2026):
    set_seed(seed)
    setup_gpu()