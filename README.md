# Speech Emotion Recognition Using Deep Learning

An end-to-end deep learning project for recognizing human emotional states from spoken audio using Log-Mel Spectrogram representations and a compact 2D Convolutional Neural Network (CNN).

---

## 1. Project Overview

Speech Emotion Recognition (SER) is a core problem in affective computing, voice-user interfaces, automated customer assistance, and psychological wellness monitoring. Human speech encodes emotional information through both acoustic prosody (pitch contours, speech rate, vocal energy) and spectral formant structures.

This project implements an end-to-end pipeline to categorize speech signals into six distinct emotional categories using a 2D CNN trained on Log-Mel Spectrogram features extracted from the **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset). The pipeline covers dataset exploration, audio standardization, time-frequency feature extraction, model development, comparative class-weighting experiments, detailed error diagnosis, and live microphone inference.

---

## 2. Key Features

- **Exploratory Data Analysis**: Granular analysis of sample counts, duration distributions, speaker demographic balance, and acoustic waveform characteristics across emotional states.
- **Audio Standardization & Preprocessing**: Duration standardization to 5.2 seconds ($83,200$ samples at 16 kHz) with right-sided zero-padding and linear 16-bit PCM normalization, preserving natural dynamic loudness across recordings with zero truncation.
- **Native TensorFlow Feature Extraction**: Pure TensorFlow signal processing (`tf.signal.stft`, `tf.signal.linear_to_mel_weight_matrix`, `tf.math.log`) extracting $(322, 128, 1)$ Log-Mel Spectrogram tensors on the fly.
- **Speaker-Independent Data Partitions**: Evaluation on strictly disjoint actor splits (train, validation, and test sets share no speakers), ensuring evaluation measures true emotion recognition rather than speaker identity recognition.
- **Compact 2D CNN Architecture**: A lightweight 3-block 2D CNN with Batch Normalization, MaxPooling, and Global Average Pooling ($28,294$ parameters, $<110\text{ KB}$) preventing overfitting.
- **Experimental Class-Weighted Comparison**: Rigorous benchmark comparing the baseline CNN against a balanced class-weighted model to address per-class recall disparities.
- **Granular Evaluation**: Confusion matrix heatmaps, classification reports, and acoustic error analysis identifying acoustic arousal-valence confusion pairs.
- **Interactive Live Microphone Inference**: Direct real-time audio capture from the user's laptop microphone with automatic hardware resampling and instant class probability predictions.

---

## 3. Supported Emotions

The system classifies audio into **six primary emotion classes**:

| Index | Emotion | CREMA-D Code | Acoustic Characteristics |
| :---: | :--- | :---: | :--- |
| **0** | **Neutral** | `NEU` | Moderate, steady pitch contours, unexaggerated energy, balanced harmonics |
| **1** | **Happy** | `HAP` | High pitch excursions, rapid vocal attacks, upward formant modulations |
| **2** | **Sad** | `SAD` | Low fundamental frequency, attenuated high harmonics, slower tempo |
| **3** | **Angry** | `ANG` | Elevated RMS intensity, sharp onsets, dense high-frequency harmonic energy |
| **4** | **Fear** | `FEA` | High pitch, fast tempo, uneven vocal tremor across mid-frequency bands |
| **5** | **Disgust** | `DIS` | Prolonged vowel sonorants, downward-sloping harmonic trajectories |

---

## 4. Dataset

This project utilizes the **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset):
- **Total Audio Clips**: Exactly 7,438 verified recordings.
- **Actors**: 91 professional actors (48 male, 43 female) of diverse ethnic backgrounds.
- **Sentences**: 12 standardized sentences spoken under diverse emotional scenarios.
- **Format**: 16,000 Hz, 1-channel mono `.wav` audio.

### Speaker-Independent Partitioning

To ensure scientific validity, the dataset is partitioned so that **no actor appears in more than one split**:

| Split | Sample Count | Percentage | Distinct Actors | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 5,144 | 69.15% | 63 actors | Parameter optimization and gradient descent |
| **Validation** | 738 | 9.92% | 9 actors | Hyperparameter tuning and EarlyStopping |
| **Test** | 1,556 | 20.92% | 19 actors | Strict out-of-sample generalization benchmark |

> **Why Speaker Independence Matters**: Deep CNNs readily memorize speaker-specific vocal signatures (timbre, vocal tract length, accent). By testing exclusively on 19 unseen actors, we evaluate whether the model captures generalized emotional prosody rather than actor identity.

### Dataset Acquisition via TensorFlow Datasets

The dataset is loaded programmatically using TensorFlow Datasets (`tfds`):
```python
import tensorflow_datasets as tfds

ds, ds_info = tfds.load(
    'crema_d',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=False
)
```
*Note: The raw dataset is not committed to this repository. When running the notebook, `tfds` automatically downloads and caches the TFRecords locally.*

---

## 5. Technology Stack

- **Programming Language**: Python 3.13.5
- **Deep Learning Framework**: TensorFlow / Keras (>= 2.21.0)
- **Dataset Pipeline**: TensorFlow Datasets (>= 4.9.10)
- **Audio Processing**: Librosa (>= 0.11.0), SoundFile (>= 0.13.0), PyDub, audioop-lts
- **Hardware Audio Capture**: SoundDevice (>= 0.5.6)
- **Data Manipulation & Visualization**: NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning Metrics**: Scikit-Learn (>= 1.5.2)
- **Environment**: Jupyter Notebook (IPython Kernel)

---

## 6. Audio Processing & Feature Extraction Pipeline

Raw time-domain audio signals exhibit variable temporal duration and phase sensitivity. They are transformed into fixed-dimension 2D Log-Mel Spectrograms:

```
[Raw 16-bit PCM Audio]
          │
          ▼
[Float32 Conversion & Scaling] (x / 32768.0 -> Range [-1.0, 1.0))
          │
          ▼
[Duration Standardization] (Right Zero-Padding / Truncation -> 83,200 samples = 5.2s)
          │
          ▼
[Short-Time Fourier Transform (STFT)] (frame_length=1024, frame_step=256, fft_length=1024)
          │
          ▼
[Magnitude Spectrogram] (|X(t, f)| -> 513 linear frequency bins)
          │
          ▼
[Mel Filterbank Mapping] (128 Mel bands from 0 Hz to 8,000 Hz)
          │
          ▼
[Logarithmic Compression] (ln(Mel + 1e-6))
          │
          ▼
[Output Feature Tensor] -> Shape: (322, 128, 1)
```

### Extraction Parameters:
- **Sampling Rate**: $16,000\text{ Hz}$
- **Target Duration**: $5.2\text{ seconds}$ ($83,200\text{ samples}$) — guarantees zero truncation across all 7,438 CREMA-D recordings (maximum observed duration is $5.01\text{s}$).
- **Normalization**: Linear division by $32768.0$, preserving genuine amplitude and loudness contrasts between subdued and intense utterances.
- **STFT Window Length**: 1,024 samples ($64.0\text{ ms}$)
- **STFT Hop Size**: 256 samples ($16.0\text{ ms}$, 75% temporal overlap)
- **Mel Filter Bins**: 128 triangular filters spanning $0\text{ Hz}$ to $8,000\text{ Hz}$ (Nyquist frequency)
- **Log Offset ($\epsilon$)**: $10^{-6}$ for numerical stability
- **Final Tensor Dimensions**: $(322\text{ time frames}, 128\text{ frequency bins}, 1\text{ channel})$

---

## 7. 2D CNN Model Architecture

The model uses a compact 2D Convolutional Neural Network designed specifically for time-frequency representations:

```
Input: (322, 128, 1)
   │
├── [Conv Block 1]: Conv2D(32, 3x3, same) ──> BatchNorm ──> ReLU ──> MaxPool2D(2x2)
│                                                                   [Output: 161, 64, 32]
├── [Conv Block 2]: Conv2D(64, 3x3, same) ──> BatchNorm ──> ReLU ──> MaxPool2D(2x2)
│                                                                   [Output: 80, 32, 64]
├── [Conv Block 3]: Conv2D(128, 3x3, same) ──> BatchNorm ──> ReLU ──> MaxPool2D(2x2)
│                                                                   [Output: 40, 16, 128]
├── [GlobalAveragePooling2D]: Averages across time and frequency axes
│                                                                   [Output: 128]
├── [Dense Layer]: 64 units, ReLU activation                        [Output: 64]
├── [Dropout]: Rate = 0.30
└── [Output Layer]: Dense(6, activation='softmax')                  [Output: 6]
```

- **Total Parameters**: $28,294$ ($110.52\text{ KB}$)
- **Trainable Parameters**: $27,846$
- **Non-trainable Parameters**: $448$ (Batch Normalization moving statistics)
- **Optimizer**: Adam ($\text{learning rate} = 10^{-3}$)
- **Loss Function**: Categorical Crossentropy

---

## 8. Experimental Results & Benchmark Comparison

Both models were trained using `EarlyStopping(patience=5, restore_best_weights=True)` and evaluated strictly on the $1,556$ held-out test recordings across $19$ unseen actors.

### Performance Summary Table

| Model Architecture | Test Loss | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 2D CNN** *(Selected)* | **1.2330** | **51.67%** | **56.13%** | **52.29%** | **50.41%** | **51.49%** |
| **Class-Weighted 2D CNN** | 1.2584 | 51.67% | 51.78% | 51.35% | 50.18% | 50.84% |

> **Model Selection Decision**: The **Baseline 2D CNN** (`models/baseline_cnn_best.keras`) is selected as the primary deployment model because it achieved a higher Macro F1-Score ($50.41\%$ vs. $50.18\%$) and lower out-of-sample test loss ($1.2330$ vs. $1.2584$).

### Per-Class Performance Breakdown (Baseline CNN)

| Emotion Class | Precision | Recall | F1-Score | Support (Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | 56.64% | 77.19% | 65.34% | 228 |
| **Happy** | 46.15% | 32.71% | 38.27% | 266 |
| **Sad** | 58.75% | 76.98% | 66.62% | 278 |
| **Angry** | 68.75% | 61.65% | 65.01% | 266 |
| **Fear** | 52.71% | 25.94% | 34.76% | 266 |
| **Disgust** | 53.57% | 38.11% | 44.50% | 252 |
| **Macro Average** | **56.13%** | **52.29%** | **50.41%** | **1,556** |
| **Weighted Average** | **56.03%** | **51.67%** | **52.29%** | **1,556** |

### Confusion Analysis:
1. **Best Recognized Emotions**: **Sadness** ($76.98\%$ recall) and **Neutral** ($77.19\%$ recall) are strongly recognized due to distinct low-arousal, flat-contour acoustic profiles. **Anger** achieved the highest precision ($68.75\%$) driven by energetic upper-harmonic signatures.
2. **Acoustic Overlap & Confusions**:
   - **Happy $\leftrightarrow$ Angry / Fear**: High-arousal emotions share rapid attacks and pitch excursions; valence (positive vs. negative) is difficult to distinguish purely from spectral envelopes.
   - **Fear $\leftrightarrow$ Neutral / Sad**: Subdued utterances with subtle pitch fluctuations create cross-talk between low/mid arousal states.

---

## 9. Live Microphone Speech Emotion Prediction

Stage 8 incorporates interactive microphone inference using `sounddevice`:

- **Direct Audio Capture**: Records $5.2\text{ seconds}$ of mono audio from the laptop microphone.
- **Dynamic Resampling**: Detects native hardware sample rates ($44.1\text{ kHz}$ or $48\text{ kHz}$) and resamples to $16,000\text{ Hz}$ via `librosa.resample`.
- **Integer PCM Scaling**: Converts normalized float32 microphone audio to 16-bit PCM integer range to maintain feature extraction consistency.
- **Instant Softmax Output**: Produces a $(1, 322, 128, 1)$ tensor and runs forward inference on `models/baseline_cnn_best.keras` in sub-second time.
- **Ranked Probabilities**: Displays all six emotion probabilities in descending order with visual confidence bars.

### Example Console Output:
```
=================================================================
LIVE MICROPHONE SPEECH EMOTION PREDICTION
=================================================================
Audio Input Device:      [2] Microphone Array (AMD Audio Device)
Hardware Sampling Rate:  44,100 Hz (Resampling target: 16,000 Hz)
Recording Window:        5.2 seconds (229,320 hardware samples)
-----------------------------------------------------------------
>>> RECORDING IN PROGRESS... Please speak now! <<<
>>> RECORDING COMPLETE. Extracting acoustic features... <<<

=================================================================
LIVE EMOTION PREDICTION RESULTS
=================================================================
Captured Waveform Samples:    83,200 (5.20s at 16,000 Hz)
Peak Signal Amplitude:        0.1377 (Float32 Range [-1.0, 1.0])
Log-Mel Feature Shape:        (322, 128, 1) (Time Frames x Mel Bins x Channels)
Batch Tensor Shape:           (1, 322, 128, 1)
-----------------------------------------------------------------
PREDICTED EMOTION:            NEUTRAL (94.71% Confidence)
-----------------------------------------------------------------
Emotion Class Probability Distribution (Descending Order):

  1. Neutral   :  94.71%  | #######################
  2. Sad       :   3.27%  | 
  3. Happy     :   0.85%  | 
  4. Disgust   :   0.55%  | 
  5. Angry     :   0.47%  | 
  6. Fear      :   0.15%  | 
=================================================================
```

> **Acoustic Disclaimer**: The predicted emotion represents an empirical estimate of vocal prosody (pitch, tempo, spectral envelope) based on actor training data. It is not an objective determination of a person's true internal psychological state.

---

## 10. Project Structure

```
Speech-Emotion-Recognition/
├── .venv/                              # Local virtual environment (excluded from Git)
├── models/
│   ├── baseline_cnn_best.keras         # Best baseline CNN model checkpoint (Primary)
│   └── class_weighted_cnn_best.keras   # Class-weighted experiment checkpoint
├── .gitignore                          # Git exclusion rules
├── README.md                           # Comprehensive documentation
├── requirements.txt                    # Pinned project dependencies
└── Speech_Emotion_Recognition.ipynb     # Complete 8-stage Jupyter Notebook
```

---

## 11. Installation & Execution Guide

### Prerequisites
- Python 3.13+ (or Python 3.11/3.12)
- Working laptop/desktop microphone
- Internet connection (first run only, for automated TFDS CREMA-D download)

### Step 1: Clone the Repository
```powershell
git clone https://github.com/vedika01105/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Launch the Jupyter Notebook
```powershell
jupyter notebook
```
Open [`Speech_Emotion_Recognition.ipynb`](Speech_Emotion_Recognition.ipynb) in your browser and execute the notebook cells sequentially from Stage 1 through Stage 8.

### Running Live Microphone Predictions
Once the notebook environment is loaded, you can record and predict live audio at any time simply by executing:
```python
predict_live_emotion()
```

---

## 12. Limitations

- **Baseline Test Accuracy**: The model achieves $51.67\%$ accuracy on $19$ completely unseen actors. While substantially above random chance ($16.67\%$), speech emotion recognition on speaker-disjoint splits remains challenging.
- **Acoustic Arousal vs. Valence Ambiguity**: High-arousal negative emotions (Anger) and high-arousal positive emotions (Happiness) exhibit similar spectral energy profiles and pitch excursions, causing occasional cross-valence misclassifications.
- **Hardware & Environmental Variance**: Laptop microphone frequency response, room acoustics, ambient fan noise, and speaking distance influence live prediction accuracy.
- **Acoustic Estimate vs. Internal State**: The model classifies acoustic prosody patterns; it does not measure true internal human affect or cognitive emotion.
- **Academic Scope**: This project is developed for educational and research exploration and is not intended for clinical diagnostic or production use.

---

## 13. Future Improvements

- **Audio Data Augmentation**: Applying SpecAugment (frequency and time masking), pitch shifting, time-stretching, and background noise injection during training.
- **Multi-Feature Fusion**: Integrating fundamental frequency ($F_0$), Chroma features, and Zero-Crossing Rate (ZCR) with Log-Mel spectrograms.
- **Temporal & Attention Architectures**: Incorporating Convolutional Recurrent Neural Networks (CRNN) with Bidirectional GRU/LSTM layers or Transformer self-attention.
- **Pretrained Self-Supervised Models**: Fine-tuning pretrained acoustic representations such as Wav2Vec 2.0 or HuBERT.
- **Multi-Corpus Benchmarking**: Evaluating generalization across additional speech corpora (RAVDESS, IEMOCAP, TESS).

---

## 14. Author

- **GitHub**: [@vedika01105](https://github.com/vedika01105)
