from src.config import set_seed
import gc
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def compile_model_with_lr(model, lr):
    optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False
    )
    return model

def lr_search(lr_list, train_ds, val_ds, build_model_fn, parameters_dict, epochs_per_trial=3, seed=2026):
    results = []

    for lr in lr_list:
        print(f"\n[INFO] Testing LR = {lr:.1e}")
        tf.keras.backend.clear_session()
        set_seed(seed)
        trial_model = build_model_fn(parameters_dict)
        trial_model = compile_model_with_lr(trial_model, lr)

        hist = trial_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_per_trial,
            verbose=1
        )

        best_val = min(hist.history["val_loss"])
        results.append({"lr": lr, "best_val_loss": best_val})
        print(f"[RESULT] LR={lr:.1e}  best_val_loss={best_val:.6f}")

        del trial_model
        gc.collect()

    print("\n===== LR SEARCH results =====")
    for r in results:
        print(r)
        
    best_lr = min(results, key=lambda x: x["best_val_loss"])["lr"]
    print("\n[INFO] Optim LR selected:", best_lr)

    return best_lr

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(
    model,
    train_ds,
    val_ds,
    epochs=100,
    checkpoint_path=None,
    plot_callback=None
):

    print("Start Training...")

    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.90,
            patience=1,
            min_lr=1e-7
        )

    callbacks = [early_stopping, reduce_lr]

    if plot_callback is not None:
        callbacks.insert(0, plot_callback)

    if checkpoint_path:
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True
        )
        callbacks.append(checkpoint)

    tic = time.time()

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=2,
        callbacks=callbacks
    )

    toc = time.time()
    training_time = toc - tic

    print(f"\nTraining finished in {training_time:.2f} seconds")

    return history, training_time