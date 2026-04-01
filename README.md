## Overview

Automatic Modulation Classification (AMC) is the task of identifying 
the modulation scheme of a received radio signal directly from raw I/Q 
samples, without prior knowledge of transmission parameters. It is a 
key component in cognitive radio, spectrum monitoring, and signal 
intelligence systems.

This repository benchmarks two deep learning architectures on the 
RadioML 2018.01A dataset — a standard AMC benchmark covering 24 
modulation types across a wide SNR range (−20 to +30 dB):

- A **patch-based Transformer**, treating the I/Q sequence as a series 
  of tokens analogous to vision patches
- A **ResNet1D**, applying residual convolutions directly on the 
  temporal signal

Both share a common modular codebase (`src/`) and training pipeline.

The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018) (`GOLD_XYZ_OSC.0001_1024.hdf5`)

---

## Dataset

| Property | Value |
|---|---|
| Dataset | RadioML 2018.01A |
| Classes | 24 modulation types |
| Input shape | (1024, 2) — I/Q samples |
| SNR range | −20 dB to +30 dB (step 2 dB) |
| Split (custom) | 70% train / 15% val / 15% test |
| Normalization | RMS per sample |
| Total samples | ~2.55 M |

<details>
<summary>Full list of 24 modulation classes</summary>

OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK

</details>

---

## Repository Structure

```
RadioML-Classifier/
├── README.md
├── requirements.txt
├── Transformers.ipynb            # Transformer training & evaluation
├── ResNet.ipynb                  # ResNet1D training & evaluation
└── src/
    ├── config.py                 # set_seed, setup_gpu, setup_environment
    ├── dataset.py                # RadioMLConfig: classes, SNR range; Functions: load_dataset, normalize_rms, split_dataset, build_tf_dataset, PyDataset (Class) 
    ├── utils.py                  # plot helpers, for dataset
    ├── train.py                  # lr_search, compile_model_with_lr, train_model
    ├── evaluate.py               # plot_loss, plot_accuracy, evaluate_model_by_snr, plot_accuracy_vs_snr, PlotLosses (during training)
    └── models/
        ├── transformers.py       # TransformerBlock, get_positional_encoding, build_transformer_model
        └── resnet.py             # residual_stack, build_resnet_model
```

> Both notebooks live at the root level and share the entire `src/` stack — only the model file differs.

---

## Models

### Transformer

A Transformer encoder with patch-based tokenization applied directly to raw I/Q sequences.

| Stage | Detail |
|---|---|
| Input | (1024, 2) |
| Patching | 64 patches × 32-dimensional tokens (patch size = 16 samples × 2 channels) |
| Projection | Linear projection to d_model = 128 |
| Positional encoding | Fixed sinusoidal positional encoding (added to tokens) |
| Encoder blocks | 2 Transformer encoder blocks (8 attention heads, FFN dim = 256) |
| Pooling | GlobalAveragePooling1D |
| Head | LayerNorm → Dense(128, SELU) → Dropout(0.2) → Dense(24, softmax) |
| **Total params** | **289,048 (1.10 MB)** |

### ResNet1D
 
A deep residual network operating directly on raw 1D I/Q signals. The number of stacks is derived automatically from the input length as `stacks = log2(1024) = 10`, so the temporal dimension is halved at each stage via MaxPooling1D until it collapses to 1 before the head.
 
**Residual stack** (repeated × 10):
 
| Layer | Detail |
|---|---|
| Conv1D 1×1 | filters = 32, linear activation — channel projection |
| Residual unit 1 | Conv1D 3 ReLU → Conv1D 3 linear → Add (skip) |
| Residual unit 2 | Conv1D 3 ReLU → Conv1D 3 linear → Add (skip) |
| MaxPooling1D | pool = 2, stride = 2 → sequence length ÷ 2 |
 
**Head:**
 
| Layer | Detail |
|---|---|
| Flatten | sequence collapses to 1 after 10 poolings → shape (32,) |
| Dense(128) | SELU + He normal init |
| Dropout(0.5) | — |
| Dense(128) | SELU + He normal init |
| Dropout(0.5) | — |
| Dense(24) | Softmax + Glorot uniform |
| **Total params** | **158K** |

---

## Training

Both models follow the same training protocol:

- **Batch size:** 256  
- **Epochs:** 100 with early stopping and ReduceLROnPlateau
- **LR selection:** grid search over `[1e-4, 3e-4, 1e-3, 3e-3]` (3 epochs/trial). 1e-3 optimal.
- **Loss:** Categorical cross-entropy  
- **Optimizer:** Adam  
- **Seed:** 2026  

---

## Results

### Overall Test Accuracy (all SNRs)

| Model | Params | Overall Accuracy |
|---|---|---|
| **Transformer** | 289K | 59.2% |
| **ResNet1D** | 158K | 61.7% |

> Overall accuracy is computed across the full SNR range (−20 to +30 dB), heavily weighted by the low-SNR regime where classification is hardest. Both models perform similarly at low SNR; ResNet1D pulls ahead significantly above 6 dB.

### Accuracy vs SNR

The Transformer plateaus at around 90–91% for SNR ≥ 10 dB. The ResNet1D keeps improving, reaching ~96–97% at SNR ≥ 10 dB — a ~6 pp gap at high SNR despite having fewer parameters.

```
SNR (dB) │ Transformer │  ResNet1D
─────────┼─────────────┼───────────
   −20   │    4.5%     │    4.2%
   −10   │   12.9%     │   14.7%
    −6   │   27.7%     │   24.7%
    −4   │   34.1%     │   33.8%
    −2   │   44.4%     │   43.8%
     0   │   56.8%     │   54.5%
    +2   │   66.2%     │   63.7%
    +4   │   76.4%     │   76.4%
    +6   │   85.0%     │   87.5%
    +8   │   88.9%     │   93.3%
   +10   │   89.8%     │   95.2%
  ≥+12   │  ~90–91%    │  ~96–97%
```

*See the Transformer and ResNet notebooks for the full `plot_accuracy_vs_snr` figures.*

### Persistent Confusion Patterns

Both models exhibit consistent confusion patterns, which reflect known structural ambiguities in the RadioML 2018.01A dataset:

- **AM-DSB-WC ↔ AM-DSB-SC** — the most challenging pair across all SNR levels; their time-domain similarity makes them difficult to distinguish, especially under noise
- **High-order QAM (128–256QAM)** — constellation density leads to overlap at moderate SNR.
- **8PSK ↔ 16PSK / 32PSK** — phase ambiguity, more pronounced in the Transformer due to its patch-based representation.

The ResNet1D confusion matrix at SNR = 12 dB is nearly diagonal across all classes, indicating stronger discriminative power from local temporal features compared to patch-based attention.

---

## Limitations

- Performance drops significantly below 0 dB SNR
- Models are trained on synthetic data (RadioML), which may not generalize to real-world signals.
  
---

## Installation

```bash
git clone https://github.com/asheredia/deep-amc-radioml.git
cd deep-amc-radioml
pip install -r requirements.txt
```

Download `GOLD_XYZ_OSC.0001_1024.hdf5` from [Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018) and update the path in the notebook:

```python
path = "/path/to/your/dataset/"
file = "GOLD_XYZ_OSC.0001_1024.hdf5"
```

Then open either notebook:

```bash
jupyter lab Transformers.ipynb
jupyter lab ResNet.ipynb
```

---

## Requirements

Tested with:
- TensorFlow 2.17 (GPU support via `tensorflow[and-cuda]`)
- Python 3.10
- NVIDIA GPU

```
tensorflow[and-cuda]>=2.17
numpy
pandas
scikit-learn
matplotlib
jupyterlab
```

---

## Hardware

Experiments were run on a single **NVIDIA RTX 6000 Ada Generation** (48 GB VRAM). Training the Transformer for 87 effective epochs took approximately **~45 minutes** at 30 s/epoch with batch size 256 on the full dataset. For the ResNet architecture, the training time was ~101 minutes at 135 s/epoch (45 effective epochs).

---

## License

MIT

---

## Citation

If you use this code, please cite the RadioML dataset:

> T. J. O’Shea, T. Roy and T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," in IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, Feb. 2018, doi: 10.1109/JSTSP.2018.2797022.
