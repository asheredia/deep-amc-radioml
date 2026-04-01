import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import clear_output

def plot_signal_analysis(sample_tuple, class_names, fs=None, n_samples=1024, use_normalized_freq=False):
    signal_array = sample_tuple[0]
    class_idx = sample_tuple[1]
    snr_db = sample_tuple[2]

    class_label = class_names[class_idx]

    # Build complex signal (I + jQ)
    complex_signal = signal_array[:, 0] + 1j * signal_array[:, 1]

    # Time axis
    if fs is None:
        time_axis = np.arange(len(complex_signal))
    else:
        time_axis = np.arange(len(complex_signal)) / fs

    # Frequency spectrum
    spectrum = np.fft.fftshift(np.fft.fft(complex_signal))

    if use_normalized_freq or fs is None:
        freq = np.fft.fftfreq(len(complex_signal), d=1.0)
        freq_shifted = np.fft.fftshift(freq)
        freq_label = "Normalized Frequency"
        freq_scale = 1.0
    else:
        freq = np.fft.fftfreq(len(complex_signal), d=1 / fs)
        freq_shifted = np.fft.fftshift(freq)
        freq_label = "Frequency [MHz]"
        freq_scale = 1e6

    # Average power
    avg_power = np.mean(np.abs(complex_signal) ** 2)
    print(f"[INFO] Average power: {avg_power:.6f}")

    # ---- Time domain plot ----
    plt.figure(figsize=(14, 9))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis[:n_samples], np.real(complex_signal[:n_samples]), label="I (real)")
    plt.plot(time_axis[:n_samples], np.imag(complex_signal[:n_samples]), label="Q (imag)", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Time Domain Signal ({n_samples} samples) | Class {class_idx} ({class_label}) | SNR {snr_db} dB")
    plt.legend()
    plt.grid()

    # ---- Frequency spectrum ----
    plt.subplot(2, 1, 2)
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    plt.plot(freq_shifted / freq_scale, magnitude_db, label="Magnitude")
    plt.xlabel(freq_label)
    plt.ylabel("Magnitude [dB]")
    plt.title("Frequency Spectrum")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # ---- Constellation diagram ----
    plt.figure(figsize=(4, 4))
    plt.scatter(np.real(complex_signal), np.imag(complex_signal), s=5, alpha=0.5, label="Samples")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("Constellation Diagram")
    plt.legend()
    plt.grid()

    plt.show()

def analyze_random_sample(sample_dict, class_names, plot_fn, fs=None, n_samples=1024):
    print(f"[INFO] Selected sample index: {sample_dict['index']}")
    print(f"[INFO] Label: {sample_dict['label']} ({class_names[sample_dict['label']]}) | SNR: {sample_dict['snr_db']} dB")

    plot_fn(
        (sample_dict["signal"], sample_dict["label"], sample_dict["snr_db"]),
        class_names,
        fs=fs,
        n_samples=n_samples
    )

def plot_class_distribution(Y, class_names):
    labels = np.argmax(Y, axis=1)
    counts = np.bincount(labels, minlength=len(class_names))

    plt.figure()
    plt.bar(class_names, counts)
    plt.title("Class Distribution")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def plot_snr_distribution(Z):
    snr_values = np.squeeze(Z)
    counts_snr = Counter(snr_values)
    snr_value = list(counts_snr.keys())
    snr_counts = list(counts_snr.values())

    plt.figure()
    #plt.hist(snr_values, bins=30)
    plt.bar(snr_value, snr_counts, width=1.5, color='green', edgecolor='black', alpha=0.7)
    plt.title("SNR Distribution")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Count")
    plt.grid()
    plt.show()


def plot_class_snr_heatmap(Y, Z, class_names):
    labels = np.argmax(Y, axis=1)
    snrs = np.squeeze(Z)

    unique_snrs = np.sort(np.unique(snrs))
    heatmap = np.zeros((len(class_names), len(unique_snrs)))

    for i, snr in enumerate(unique_snrs):
        mask = snrs == snr
        counts = np.bincount(labels[mask], minlength=len(class_names))
        heatmap[:, i] = counts

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto')
    plt.colorbar(label="Count")
    plt.xticks(range(len(unique_snrs)), unique_snrs)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Class")
    plt.title("Class vs SNR Distribution")
    plt.show()

def plot_all_classes_at_snr(X, Y, Z, class_names, snr=10):
    labels = np.argmax(Y, axis=1)
    snrs = np.squeeze(Z)
    n_classes = len(class_names)
    plt.figure(figsize=(12, 2.5 * n_classes))

    for class_idx, class_name in enumerate(class_names):
        idxs = np.where((labels == class_idx) & (snrs == snr))[0]
        if len(idxs) == 0:
            continue

        sample = X[idxs[0]]
        signal = sample[:, 0] + 1j * sample[:, 1]

        plt.subplot(len(class_names), 1, class_idx + 1)
        plt.plot(np.real(signal), label="I")
        plt.plot(np.imag(signal), label="Q", alpha=0.7)
        plt.title(f"{class_name} @ {snr} dB")
        plt.grid()

    plt.tight_layout()
    plt.show()


def plot_class_across_snrs(X, Y, Z, class_names, class_name="BPSK"):
    labels = np.argmax(Y, axis=1)
    snrs = np.squeeze(Z)

    class_idx = class_names.index(class_name)
    target_snrs = [-20, -10, 0, 10, 20]

    plt.figure(figsize=(12, 8))

    for i, snr in enumerate(target_snrs):
        idxs = np.where((labels == class_idx) & (snrs == snr))[0]
        if len(idxs) == 0:
            continue

        sample = X[idxs[0]]
        signal = sample[:, 0] + 1j * sample[:, 1]

        plt.subplot(len(target_snrs), 1, i + 1)
        plt.plot(np.real(signal), label="I")
        plt.plot(np.imag(signal), label="Q", alpha=0.7)
        plt.title(f"{class_name} @ {snr} dB")
        plt.grid()

    plt.tight_layout()
    plt.show()

def plot_spectrograms_by_class(X, Y, Z, class_names, snr=10):
    labels = np.argmax(Y, axis=1)
    snrs = np.squeeze(Z)
    n_classes = len(class_names)
    plt.figure(figsize=(12, 2.5 * n_classes))

    for class_idx, class_name in enumerate(class_names):
        idxs = np.where((labels == class_idx) & (snrs == snr))[0]
        if len(idxs) == 0:
            continue

        sample = X[idxs[0]]
        signal = sample[:, 0] + 1j * sample[:, 1]

        plt.subplot(len(class_names), 1, class_idx + 1)
        plt.specgram(signal, NFFT=128, noverlap=96, Fs=1.0)
        plt.title(f"{class_name} @ {snr} dB")

    plt.tight_layout()
    plt.show()

def plot_constellation_grid(X, Y, Z, class_names, snr=10):

    labels = np.argmax(Y, axis=1)
    snrs = np.squeeze(Z)

    fig, axes = plt.subplots(6, 4, figsize=(14, 18))
    axes = axes.ravel()

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        idxs = np.where((labels == class_idx) & (snrs == snr))[0]
        
        if len(idxs) == 0:
            ax.set_title(f"{class_name}\n(no data @ {snr} dB)", fontsize=8)
            ax.axis("off")
            continue

        signal = X[idxs[0]]

        ax.scatter(signal[:, 0], signal[:, 1], s=1, alpha=0.5)

        ax.set_title(class_name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

    for i in range(len(class_names), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Constellation Grid", fontsize=14)
    plt.tight_layout()
    plt.show()
