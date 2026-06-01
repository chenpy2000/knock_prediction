# Engine Knock Prediction with a Transformer

> Forecasting the cylinder **knock-sensor signal** of a diesel–CNG dual-fuel engine, and deriving knock events directly from the forecast.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8%20%7C%20CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Overview

An encoder-only Transformer reads a window of past knock-sensor samples — serialized per crank angle, per cycle into one long time series — and forecasts the next stretch of the signal. Knock is then **derived from the forecast** using a band-pass + amplitude-threshold (MAPO-style) rule and scored against ground-truth labels. The full pipeline and the quantitative analysis live in [`knock_prediction.ipynb`](knock_prediction.ipynb).

## Why it matters

With sequence models like Transformers, an internal combustion engine could anticipate its working condition before it happens. That would let control actuators — valve/injection timing, fuel injection, ignition — trigger dynamically from the *predicted* state rather than reacting after the fact, with strong potential to push thermal efficiency higher.

## Dataset

| | |
|---|---|
| **Source** | Mukhsin, Muammar (2021), *"Diesel-CNG Dual Fuel Engine Knock Signal at 1400 rpm with Various Diesel to CNG Fuel Ratios"*, Mendeley Data, V1 |
| **DOI** | [10.17632/jzmhw4mbn7.1](https://doi.org/10.17632/jzmhw4mbn7.1) |
| **Signal** | Piezoelectric knock-sensor voltage (Volt) |
| **Engine speed** | 1400 rpm |
| **Blends** | `100D`, `90D10G`, `80D20G`, `70D30G`, `60D40G` (diesel-to-CNG ratios) |
| **Per blend** | 360 crank angles (-180°…+179°) × 500 cycles → **180,000 samples** once serialized (1 sample / crank-angle degree, *f*ₛ = 8400 Hz) |

## Method

The notebook runs end-to-end:

1. **Data loading & preprocessing** — read one fuel-blend sheet, serialize it *column-major* so cycles sit end-to-end in time order, then label knock per sample by band-pass filtering (1200–4000 Hz) to isolate the high-frequency ringing and flagging amplitudes that exceed `k·σ` of the baseline noise, restricted to the post-TDC knock window.
2. **Inspection** — plot a slice of the serialized signal with detected knock points overlaid.
3. **Windowed dataset** — sliding windows where `x` = `INPUT_LEN` past steps of `[normalized signal, knock flag]` and `y` = the next `OUTPUT_LEN` steps of the signal. The split is **temporal** (earliest part → train, rest → val) and reuses the *training* mean/std on validation to avoid leakage.
4. **Model** — an encoder-only Transformer with sinusoidal positional encoding; it encodes the window, **mean-pools** over time, and a small MLP head projects to all `OUTPUT_LEN` future steps at once (direct multi-step forecasting, no autoregressive decoding). **201,808 parameters**.
5. **Training** — Adam + MSE loss over the temporal train/validation split.
6. **Knock prediction & evaluation** — the predicted horizon is band-pass filtered and thresholded with the same MAPO-style rule used for the labels; reports **signal** metrics (MSE, MAE, R²) and **knock** classification metrics.
7. **Quantitative analysis** — dataset characterization, knock-vs-blend statistics, spectral analysis, and naive forecasting baselines for context (Sections A–E in the notebook).

## Results

**Setup.** `BLEND = 90D10G`, context `INPUT_LEN = 1440` → forecast `OUTPUT_LEN = 720` (one full four-stroke cycle ahead). Trained 100 epochs (Adam, lr 1e-3, MSE) on an 80/20 temporal split; validated on 8,461 sliding windows. Run on PyTorch 2.8.0 + CUDA 12.8 (GPU).

**Training.** Train and validation MSE descend together with no overfitting gap; best validation MSE (normalized) **0.1338 at epoch 99**.

![Training curve](training_curve.png)

**Signal forecasting** — validation, de-normalized (raw Volt) scale, over 8,461 × 720 = 6,091,920 samples:

| Model | MSE | MAE | R² |
|---|---|---|---|
| persistence (repeat last sample) | 0.3036 | 0.4432 | -0.99 |
| train-mean | 0.1524 | 0.3168 | ≈0.00 |
| seasonal-naive (repeat last 360-sample cycle) | 0.0684 | 0.2077 | 0.55 |
| **Transformer** | **0.0212** | **0.1122** | **0.86** |

The Transformer cuts MSE ~3.2× below the strongest naive baseline (seasonal-naive) and reaches **R² ≈ 0.86** — it is modeling the waveform, not just its periodicity. A representative validation window (predicted vs. true over the 720-step horizon):

![Forecast](forecast.png)

Per-horizon error makes the gap to the baselines explicit across the forecast:

![Per-horizon error](horizon_error.png)

**Knock detection (derived from the forecast)** — stated plainly, this is where the current pipeline falls short:

| accuracy | precision | recall | F1 |
|---|---|---|---|
| 0.9952 | 0.000 | 0.000 | 0.000 |

The 99.5% accuracy is an artifact of class imbalance: knock is only ~0.48% of validation samples (29,506 of 6,091,920), and the model predicts **zero** knock events (TP = 0, FP = 0). Band-pass + threshold on a smooth multi-step forecast suppresses exactly the high-frequency ringing the detector keys on, so recall collapses to 0. **In its current form the model is a strong signal forecaster but not a knock detector** — see Limitations for concrete directions.

## Dataset analysis

From the analysis sections of the notebook (same detector as the pipeline, so numbers are comparable):

**Knock vs. fuel blend.** Knock rate is non-monotonic in CNG fraction — it falls through the mid blends, then jumps sharply at 40% CNG, where 96% of cycles knock:

| blend | CNG % | knock rate % | knocking cycles % | mean MAPO (V) |
|---|---|---|---|---|
| 100D | 0 | 0.40 | 71.6 | 0.185 |
| 90D10G | 10 | 0.45 | 73.2 | 0.183 |
| 80D20G | 20 | 0.23 | 45.6 | 0.153 |
| 70D30G | 30 | 0.10 | 22.6 | 0.131 |
| 60D40G | 40 | **1.68** | **96.0** | **0.279** |

![Knock vs CNG](knock_vs_cng.png)

The 40% spike is robust across detector thresholds (`k` = 3…6), so it reflects the data rather than the threshold. Two further observations: **cycle-to-cycle variability** (COV of the in-window peak) *decreases* with CNG, from 18.6% (100D) to 11.6% (60D40G); and **knock-band energy** (1200–4000 Hz) is ~1.1–1.6% of total signal energy per blend.

## Repository structure

```
knock_prediction/
├── knock_prediction.ipynb       # full pipeline + quantitative analysis
├── MethodsX Data 1400 rpm.xlsx  # source dataset (5 blend sheets)
├── requirements.txt             # dependencies
├── forecast.png                 # forecast vs. ground truth (validation window)
├── training_curve.png           # train/val MSE over 100 epochs
├── horizon_error.png            # per-horizon MSE: Transformer vs. baselines
└── knock_vs_cng.png             # knock rate & MAPO vs. CNG fraction
```

## Setup

```bash
git clone https://github.com/chenpy2000/knock_prediction.git
cd knock_prediction
pip install -r requirements.txt
jupyter lab        # open knock_prediction.ipynb and run top to bottom
```

> Run on PyTorch 2.8.0 (CUDA 12.8) on GPU. `requirements.txt` currently pins a different build (`torch==2.11.0+cu126`); install the wheel matching your CUDA/CPU setup, and reconcile the pin before release.

## Configuration

All knobs live in one config cell near the top of the notebook:

| Data / training | | Model | |
|---|---|---|---|
| `BLEND` | `90D10G` | `D_MODEL` | 64 |
| `INPUT_LEN` | 1440 | `N_HEAD` | 1 |
| `OUTPUT_LEN` | 720 | `N_LAYERS` | 3 |
| `STRIDE` | 4 | `DIM_FF` | 128 |
| `BATCH_SIZE` | 256 | `DROPOUT` | 0.1 |
| `EPOCHS` | 100 | `IN_CHANNELS` | 2 (signal, knock) |
| `LR` | 1e-3 | `OUT_CHANNELS` | 1 (signal) |
| `VAL_FRAC` | 0.2 | **params** | **201,808** |

## Limitations & next steps

- **Knock recall is the core open problem.** The forecast tracks the signal (R² ≈ 0.86) but recovers no knock events (recall 0). Promising fixes: (a) predict knock with a dedicated classification head instead of re-deriving it from a smoothed forecast; (b) handle the ~0.5% class imbalance (weighted/focal loss, resampling); (c) forecast the band-passed knock envelope as an auxiliary target; (d) calibrate the detection threshold on predictions via a precision–recall sweep.
- **Single blend, single speed.** Results are for `90D10G` at 1400 rpm; cross-blend and cross-speed generalization remain open (the notebook includes a cross-blend scaffold).
- **Odd head count warning.** `N_HEAD = 1` triggers a benign PyTorch `enable_nested_tensor` warning during training; it does not affect results.

## Citation

```bibtex
@misc{mukhsin2021knock,
  author       = {Mukhsin, Muammar},
  title        = {Diesel-CNG Dual Fuel Engine Knock Signal at 1400 rpm
                  with Various Diesel to CNG Fuel Ratios},
  year         = {2021},
  publisher    = {Mendeley Data},
  version      = {V1},
  doi          = {10.17632/jzmhw4mbn7.1}
}
```
