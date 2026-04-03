# RadioML-Classifier
 
This is a Deep learning benchmark for **Automatic Modulation Classification (AMC)** on the [RadioML 2018.01A](https://www.kaggle.com/datasets/pinxau1000/radioml2018) dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`). Six architectures are evaluated and compared: from CNN and LSTM architectures to a patch-based Transformer, sharing the same modular codebase.
 
## Overview
 
The aim of Automatic Modulation Classification (AMC) is to identify the modulation scheme of a received radio signal directly from raw In-phase and Quadrature (I/Q) samples, without prior knowledge of transmission parameters. It is a key component in cognitive radio, spectrum monitoring, and signal intelligence systems.
 
This repository benchmarks six deep learning architectures on RadioML 2018.01A (from DeepSig), a standard AMC benchmark covering 24 modulation types across a wide SNR range (−20 to +30 dB), analyzing the trade-off between classification accuracy and computational efficiency.
 
---
 
## Dataset
 
| Property | Value |
|---|---|
| Dataset | RadioML 2018.01A |
| Classes | 24 modulation types |
| Input shape | (1024, 2) — I/Q samples |
| SNR range | −20 dB to +30 dB (step 2 dB) |
| Split | 70% train / 15% val / 15% test (stratified by class) |
| Normalization | RMS per sample (applied, not provided by dataset) |
| Total samples | ~2.55 M |

> **Some preprocessing notes:**
> - **Split:** The dataset is distributed as a single HDF5 file, so it does not contain a 
>   predefined train/val/test partition. The 70/15/15 split is applied via 
>   a helper function.
> - **Normalization:** RMS normalization is not provided by the dataset. It is 
>   applied per sample during loading in order 
>   to remove amplitude dependence across SNR levels, making the model to focus in modulation structure rather than 
>   signal power.

<details>
<summary>Full list of 24 modulation classes (click to see)</summary>
 
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
├── Baselines.ipynb               # CNN, LSTM, CNN+LSTM, CNN+GRU training & evaluation
├── Comparison.ipynb              # Cross-model accuracy vs SNR + efficiency benchmark
├── results/
│   ├── results_cnn_acc.npy
│   ├── results_cnn_lstm_acc.npy
│   ├── results_lstm_acc.npy
│   ├── results_cnn_gru_acc.npy
│   ├── results_resnet_acc.npy
│   ├── results_transformers_acc.npy
│   └── models_measurements.csv  # MFLOPs, latency, throughput per model
└── src/
    ├── config.py                 # set_seed, setup_gpu, setup_environment
    ├── dataset.py                # RadioMLConfig: classes, SNR range; load_dataset,
    │                             # normalize_rms, split_dataset, build_tf_dataset, PyDataset
    ├── utils.py                  # plot helpers
    ├── train.py                  # lr_search, compile_model_with_lr, train_model
    ├── evaluate.py               # evaluate_model_by_snr, plot_accuracy_vs_snr, PlotLosses
    └── models/
        ├── transformers.py       # TransformerBlock, get_positional_encoding, build_transformer_model
        ├── resnet.py             # residual_stack, build_resnet_model
        └── baselines.py          # cnn_model, lstm_model, cnn_lstm_model, cnn_gru_model
```
 
> All notebooks live at the root level and share the entire `src/` stack — only the model file differs. Pre-computed accuracy results (`.npy`) and efficiency metrics (`.csv`) are included in `results/`.

---
 
## Models
 
### Transformer
 
A patch-based Transformer encoder applied directly to raw I/Q sequences, inspired by Vision Transformers (ViT).
 
| Stage | Detail |
|---|---|
| Input | (1024, 2) |
| Patching | 64 patches × 32-dimensional tokens (patch size = 16 samples × 2 channels) |
| Projection | Linear projection to d_model = 128 |
| Positional encoding | Fixed sinusoidal encoding added to tokens |
| Encoder blocks | 2 × TransformerBlock (8 attention heads, FFN dim = 256) |
| Pooling | GlobalAveragePooling1D |
| Head | LayerNorm → Dense(128, SELU) → Dropout(0.2) → Dense(24, softmax) |
| **Total params** | **289K** |
 
### ResNet1D
 
A deep residual network operating directly on raw 1D I/Q signals. The number of stacks is derived automatically from the input length as `stacks = log2(1024) = 10`, halving the temporal dimension at each stage until it collapses to 1 before the head.
 
**Residual stack** (repeated × 10):
 
| Layer | Detail |
|---|---|
| Conv1D 1×1 | filters = 32, linear — channel projection |
| Residual unit 1 | Conv1D(3) ReLU → Conv1D(3) linear → Add (skip) |
| Residual unit 2 | Conv1D(3) ReLU → Conv1D(3) linear → Add (skip) |
| MaxPooling1D | pool = 2 → sequence length ÷ 2 |
 
**Head:** Flatten → Dense(128, SELU) → Dropout(0.5) → Dense(128, SELU) → Dropout(0.5) → Dense(24, softmax)
 
| **Total params** | **158K** |
|---|---|
 
### Baselines
 
| Model | Architecture | Params |
|---|---|---|
| **CNN** | 3 × [Conv1D → BN → MaxPool] → GAP → Dense(128) → Dense(24) | 161K |
| **CNN+LSTM** | 3 × [Conv1D → MaxPool] → 2 × LSTM → Dense(128) → Dense(24) | 267K |
| **LSTM** | 3 × LSTM(128/128/64) → Dense(128) → Dense(24) | 259K |
| **CNN+GRU** | 2 × [Conv1D → BN → MaxPool] → 2 × GRU → Dense(128) → Dense(24) | 174K |
 
---
 
## Training
 
All models follow the same training protocol:
 
- **Batch size:** 256
- **Epochs:** 100 with ReduceLROnPlateau
- **LR selection:** grid search over `[1e-4, 3e-4, 1e-3, 3e-3]` (3 epochs/trial) — optimal: `1e-3`
- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam
- **Seed:** 2026
 
---
 
## Results
 
### Efficiency Comparison
 
| Model | Params | MFLOPs | Latency (ms) | Throughput (samples/s) | Recurrent |
|---|---|---|---|---|---|
| CNN | 161K | 77.17 | 0.77 | 1294 | No |
| CNN+LSTM | 267K | 118.48 | 4.45 | 225 | Yes |
| LSTM | 259K | 252.70 | 28.20 | 35 | Yes |
| CNN+GRU | 174K | 88.86 | 7.65 | 131 | Yes |
| ResNet1D | 158K | 53.07 | 2.88 | 347 | No |
| **Transformer** | **289K** | **34.98** | **1.48** | **671** | **No** |
 
> The Transformer achieves the lowest MFLOPs and second-highest throughput despite having the most parameters. It reflects a direct consequence of its non-recurrent and fully parallelizable architecture.
 
### Overall Test Accuracy (all SNRs)
 
| Model | Overall Accuracy |
|---|---|
| LSTM | 65.8% |
| CNN+GRU | 63.5% |
| CNN+LSTM | 63.3% |
| ResNet1D | 61.7% |
| Transformer | 59.2% |
| CNN | 57.9% |
 
> Overall accuracy is averaged across the full SNR range (−20 to +30 dB), highly influenced by the low-SNR regime where all models perform similarly.
 
### Accuracy vs SNR
 
```
SNR (dB) │  CNN  │ CNN+LSTM │  LSTM  │ CNN+GRU │ ResNet1D │ Transformer
─────────┼───────┼──────────┼────────┼─────────┼──────────┼────────────
   −20   │  4.5% │   4.5%   │  4.5%  │   4.2%  │   4.3%   │    4.5%
   −18   │  4.1% │   4.4%   │  4.6%  │   4.3%  │   4.3%   │    4.2%
   −16   │  4.5% │   5.2%   │  5.5%  │   5.1%  │   5.2%   │    4.9%
   −14   │  5.5% │   6.3%   │  7.1%  │   6.3%  │   6.4%   │    6.1%
   −12   │  7.6% │   9.8%   │ 11.0%  │   9.8%  │   9.6%   │    8.5%
   −10   │ 10.7% │  14.8%   │ 15.6%  │  14.5%  │  14.7%   │   12.9%
    −8   │ 17.7% │  20.2%   │ 20.9%  │  20.4%  │  19.0%   │   20.0%
    −6   │ 26.2% │  27.3%   │ 28.0%  │  27.0%  │  24.7%   │   27.7%
    −4   │ 33.1% │  34.9%   │ 36.2%  │  34.8%  │  33.8%   │   34.1%
    −2   │ 41.5% │  45.2%   │ 47.7%  │  44.3%  │  43.8%   │   44.4%
     0   │ 51.8% │  55.8%   │ 58.5%  │  54.9%  │  54.5%   │   56.8%
    +2   │ 60.1% │  65.9%   │ 71.2%  │  65.2%  │  63.7%   │   66.2%
    +4   │ 70.0% │  78.0%   │ 84.5%  │  78.7%  │  76.4%   │   76.4%
    +6   │ 79.5% │  88.1%   │ 93.7%  │  88.7%  │  87.5%   │   85.0%
    +8   │ 84.9% │  93.2%   │ 96.2%  │  93.3%  │  93.4%   │   88.9%
   +10   │ 88.5% │  95.0%   │ 96.9%  │  94.8%  │  95.2%   │   89.8%
   +12   │ 90.2% │  95.8%   │ 97.0%  │  95.5%  │  96.2%   │   90.8%
   +14   │ 90.8% │  95.9%   │ 97.2%  │  95.7%  │  96.4%   │   90.6%
   +16   │ 91.2% │  96.3%   │ 97.3%  │  95.9%  │  96.6%   │   90.7%
   +18   │ 91.2% │  96.1%   │ 97.0%  │  95.8%  │  96.5%   │   90.5%
   +20   │ 91.4% │  96.0%   │ 96.9%  │  95.7%  │  96.2%   │   90.3%
   +22   │ 91.3% │  96.0%   │ 97.3%  │  95.9%  │  96.7%   │   90.8%
   +24   │ 91.2% │  96.2%   │ 97.1%  │  95.8%  │  96.6%   │   90.8%
   +26   │ 91.5% │  96.2%   │ 97.2%  │  96.0%  │  96.9%   │   90.8%
   +28   │ 91.3% │  96.4%   │ 97.4%  │  96.3%  │  96.8%   │   90.7%
   +30   │ 91.4% │  96.3%   │ 97.2%  │  95.9%  │  96.7%   │   90.7%
```
 
*See notebooks for the full `plot_accuracy_vs_snr` figure with all 6 models.*
 
### Key Findings
 
**At low SNR (≤ −6 dB):** all models converge to similar accuracy (~4–28%). The classification problem is dominated by noise and architecture choice has minimal impact.
 
**In the transition region (−4 to +6 dB):** LSTM-based models stand out, benefiting from explicit sequential modeling of inter-symbol dependencies. The gap is more noticeable above 0 dB.
 
**At high SNR (≥ +10 dB):** LSTM, CNN+LSTM, CNN+GRU and ResNet1D saturate around 96–97%. The CNN and Transformer plateau at ~91% — the Transformer's patch-based tokenization fragments inter-symbol phase transitions, limiting discrimination of high-order PSK schemes (16PSK, 32PSK).
 
**Efficiency vs. accuracy trade-off:** The Transformer offers the best accuracy/compute ratio for latency-constrained scenarios (1.48 ms, 671 samples/s, 34.98 MFLOPs). LSTM achieves the highest accuracy but at 28.2 ms latency (19× slower than the Transformer) making it unsuitable for real-time deployment.
 
### Persistent Confusion Patterns
 
Across all models and SNR levels:
 
- **AM-DSB-WC ↔ AM-DSB-SC** — the hardest pair (as well as the SSB versions); time-domain similarity makes them nearly indistinguishable under noise
- **High-order QAM (128–256QAM)** — constellation density causes overlap at moderate SNR
- **8PSK ↔ 16PSK / 32PSK** — phase boundary ambiguity, more pronounced in the Transformer due to patch-based tokenization
 
---
 
## Limitations
 
- Performance drops significantly below 0 dB SNR across all architectures
- Models are trained and evaluated on RadioML 2018.01A, where each example 
  is generated with independently drawn random channel parameters (frequency 
  offset, phase noise, SNR, etc.). Generalization to real-world hardware 
  impairments beyond those modeled in the dataset is not guaranteed.
 
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
 
Then open any notebook:
 
```bash
jupyter lab Transformers.ipynb
jupyter lab ResNet.ipynb
jupyter lab Baselines.ipynb
```
 
---
 
## Requirements
 
Tested with Python 3.10 and TensorFlow 2.17 (GPU).
 
```
tensorflow[and-cuda]>=2.17
numpy
pandas
scikit-learn
matplotlib
jupyterlab
h5py
```
 
---
 
## Hardware
 
All experiments were run on a single **NVIDIA RTX 6000 Ada Generation** (48 GB VRAM).
 
| Model | Effective Epochs | Time/Epoch | Total Training Time |
|---|---|---|---|
| Transformer | 87 | ~30 s | ~45 min |
| ResNet1D | 45 | ~135 s | ~101 min |
| CNN | 49 | ~108 s | ~88 min |
| LSTM | 72 | ~551 s | ~662 min |
| CNN+LSTM | 32 | ~97 s | ~52 min |
| CNN+GRU | 79 | ~193 s | ~255 min |
 
---
 
## License
 
MIT
 
---

 
## Citation
 
If you use this code, please cite the RadioML dataset:
 
> T. J. O'Shea, T. Roy and T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," in IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, Feb. 2018, doi: 10.1109/JSTSP.2018.2797022.
