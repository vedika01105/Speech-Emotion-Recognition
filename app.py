import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import io

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Dark Navy & Purple Theme CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Navy Background & Global Typography */
    .stApp {
        background-color: #0A0F1D;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Headers & Subtitles */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Header Gradient Title */
    .gradient-title {
        background: linear-gradient(135deg, #A855F7 0%, #6366F1 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.75rem;
    }
    
    /* Custom Card Containers */
    .card-container {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.25rem;
    }
    
    /* Prominent Prediction Box */
    .prediction-card {
        background: linear-gradient(145deg, #161F37 0%, #1A173B 100%);
        border: 1.5px solid #8B5CF6;
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        box-shadow: 0 10px 30px -5px rgba(139, 92, 246, 0.25);
        margin: 1.25rem 0;
    }
    
    .predicted-emotion-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin: 0.5rem 0;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        color: #C084FC;
        border: 1px solid #A855F7;
        border-radius: 20px;
        padding: 0.35rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    .source-badge {
        display: inline-block;
        background: #1E293B;
        color: #38BDF8;
        border-radius: 8px;
        padding: 0.2rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 1.75rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #6D28D9 0%, #4338CA 100%);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.55);
        transform: translateY(-1px);
    }
    
    /* File Uploader and Audio Input Area */
    div[data-testid="stFileUploader"] {
        background-color: #111827;
        border: 1px dashed #3B82F6;
        border-radius: 12px;
        padding: 1rem;
    }
    
    div[data-testid="stAudioInput"] {
        background-color: #111827;
        border: 1px dashed #8B5CF6;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        margin-bottom: 0.75rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px;
        border: 1px solid #1E293B;
        color: #94A3B8;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        border-color: #475569;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        border-color: #8B5CF6 !important;
        color: #C084FC !important;
    }
    
    /* Metric Card */
    .metric-box {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Model Preprocessing Parameters (Exact Notebook Specs)
# ---------------------------------------------------------
SAMPLE_RATE = 16000
TARGET_DURATION = 5.2  # seconds
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)  # 83,200 samples
PCM_NORMALIZATION_FACTOR = 32768.0

FRAME_LENGTH = 1024
FRAME_STEP = 256
FFT_LENGTH = 1024
NUM_MEL_BANDS = 128
LOWER_FREQ_HZ = 0.0
UPPER_FREQ_HZ = 8000.0
EPSILON = 1e-6

# Class Labels & Visual Styling
EMOTION_CLASSES = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']

EMOTION_META = {
    'Neutral': {'emoji': '😐', 'color': '#64748B', 'accent': '#94A3B8'},
    'Happy':   {'emoji': '😊', 'color': '#EAB308', 'accent': '#FDE047'},
    'Sad':     {'emoji': '😢', 'color': '#38BDF8', 'accent': '#7DD3FC'},
    'Angry':   {'emoji': '😡', 'color': '#EF4444', 'accent': '#F87171'},
    'Fear':    {'emoji': '😨', 'color': '#A855F7', 'accent': '#C084FC'},
    'Disgust': {'emoji': '🤢', 'color': '#10B981', 'accent': '#34D399'}
}

# ---------------------------------------------------------
# Cached Resource: Model & Precomputed Filterbank
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing trained baseline CNN & signal pipeline...")
def load_ser_model():
    """
    Loads the trained baseline CNN model and precomputes the Linear-to-Mel matrix.
    Loaded strictly once at startup.
    """
    model_path = os.path.join("models", "baseline_cnn_best.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    num_spectrogram_bins = FFT_LENGTH // 2 + 1  # 513 bins
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BANDS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_FREQ_HZ,
        upper_edge_hertz=UPPER_FREQ_HZ
    )
    return model, mel_weights

try:
    model, mel_weight_matrix = load_ser_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------
# Signal Preprocessing Logic (Reproducing Notebook Pipeline)
# ---------------------------------------------------------
def process_audio_file(file_bytes):
    """
    Decodes audio (WAV/MP3/Mic Bytes), converts to mono 16 kHz float32,
    scales to 16-bit PCM scale, standardizes duration to 5.2s,
    and extracts the (322, 128, 1) Log-Mel spectrogram.
    """
    # 1. Load audio with Librosa (supports WAV, MP3, and browser audio streams)
    y, orig_sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    if len(y) == 0:
        raise ValueError("Audio recording is empty. Please provide an active speech sample.")
    orig_duration = len(y) / orig_sr
    
    # 2. Resample to 16,000 Hz if needed
    if orig_sr != SAMPLE_RATE:
        y_16k = librosa.resample(y, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
    else:
        y_16k = y
    
    # 3. Scale float32 to signed 16-bit integer PCM
    audio_pcm = np.clip(y_16k * 32767.0, -32768.0, 32767.0).astype(np.int16)
    
    # 4. Normalize by 32768.0 (exact training preprocessing)
    audio_float = tf.cast(audio_pcm, tf.float32) / PCM_NORMALIZATION_FACTOR
    
    # 5. Standardize length to 83,200 samples via post-padding with zeros or truncation
    pad_len = tf.maximum(0, TARGET_SAMPLES - tf.shape(audio_float)[0])
    audio_padded = tf.pad(audio_float, [[0, pad_len]], constant_values=0.0)
    audio_std = audio_padded[:TARGET_SAMPLES]
    audio_std.set_shape([TARGET_SAMPLES])
    
    # 6. Extract Log-Mel Spectrogram via TensorFlow ops
    stft_out = tf.signal.stft(
        audio_std,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        pad_end=False
    )
    mag_spec = tf.abs(stft_out)
    mel_spec = tf.matmul(mag_spec, mel_weight_matrix)
    log_mel = tf.math.log(mel_spec + EPSILON)
    feature = tf.expand_dims(log_mel, axis=-1)  # (322, 128, 1)
    
    return y_16k, orig_sr, orig_duration, audio_std.numpy(), log_mel.numpy(), feature

# ---------------------------------------------------------
# Reusable Modular Visualizations & Cards
# ---------------------------------------------------------
def render_prediction_card(pred_emotion, confidence, source_name):
    meta = EMOTION_META[pred_emotion]
    st.markdown(f"""
    <div class="prediction-card">
        <span style="font-size: 3rem;">{meta['emoji']}</span>
        <div class="predicted-emotion-title" style="color: {meta['accent']};">
            {pred_emotion.upper()}
        </div>
        <div class="confidence-badge">
            Confidence: {confidence * 100:.2f}%
        </div>
        <br>
        <div class="source-badge">
            Target: {source_name}
        </div>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.75rem; margin-bottom: 0;">
            Dominant acoustic pattern identified across vocal prosody & Mel frequencies.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_chart(probs, pred_emotion):
    meta = EMOTION_META[pred_emotion]
    confidence = float(np.max(probs))
    
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor='#0A0F1D')
    ax.set_facecolor('#111827')
    
    y_pos = np.arange(len(EMOTION_CLASSES))
    bar_colors = [
        meta['accent'] if emo == pred_emotion else '#475569'
        for emo in EMOTION_CLASSES
    ]
    
    bars = ax.barh(y_pos, probs * 100, color=bar_colors, height=0.55, edgecolor='#1E293B', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{EMOTION_META[e]['emoji']} {e}" for e in EMOTION_CLASSES],
        fontsize=10,
        fontweight='600',
        color='#F1F5F9'
    )
    ax.invert_yaxis()
    ax.set_xlabel("Confidence / Softmax Probability (%)", fontsize=9, color='#94A3B8', labelpad=8)
    ax.set_xlim(0, 105)
    
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.2f}%",
            va='center',
            ha='left',
            fontsize=9,
            fontweight='700' if prob == confidence else '500',
            color='#F8FAFC' if prob == confidence else '#94A3B8'
        )
    
    for spine in ax.spines.values():
        spine.set_color('#1E293B')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#CBD5E1')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_acoustic_visualizations(audio_std, log_mel, source_name):
    st.markdown("---")
    st.markdown(f"### 📈 Acoustic Visualizations &nbsp;<span style='color: #94A3B8; font-size: 0.9rem;'>({source_name})</span>", unsafe_allow_html=True)
    
    tab_wave, tab_spec = st.tabs(["🌊 Time-Domain Waveform", "🌈 Log-Mel Spectrogram"])
    
    with tab_wave:
        fig_wave, ax_wave = plt.subplots(figsize=(12, 3), facecolor='#0A0F1D')
        ax_wave.set_facecolor('#111827')
        
        time_axis = np.linspace(0, TARGET_DURATION, TARGET_SAMPLES)
        ax_wave.plot(time_axis, audio_std, color='#38BDF8', linewidth=0.8, alpha=0.9)
        ax_wave.axhline(0, color='#334155', linestyle='--', linewidth=0.7)
        
        ax_wave.set_title(f"Standardized Waveform (5.2s @ 16 kHz | {TARGET_SAMPLES:,} samples)", color='#F8FAFC', fontsize=11, fontweight='600', pad=10)
        ax_wave.set_xlabel("Time (seconds)", color='#94A3B8', fontsize=9)
        ax_wave.set_ylabel("Normalized Amplitude", color='#94A3B8', fontsize=9)
        ax_wave.set_xlim(0, TARGET_DURATION)
        ax_wave.set_ylim(-1.05, 1.05)
        
        for spine in ax_wave.spines.values():
            spine.set_color('#1E293B')
        ax_wave.tick_params(colors='#94A3B8', labelsize=8)
        ax_wave.grid(linestyle=':', alpha=0.2, color='#64748B')
        plt.tight_layout()
        st.pyplot(fig_wave, use_container_width=True)
        plt.close(fig_wave)
        
    with tab_spec:
        fig_spec, ax_spec = plt.subplots(figsize=(12, 3.2), facecolor='#0A0F1D')
        ax_spec.set_facecolor('#111827')
        
        im = ax_spec.imshow(
            log_mel.T,
            origin='lower',
            aspect='auto',
            extent=[0, TARGET_DURATION, 0, NUM_MEL_BANDS],
            cmap='magma'
        )
        
        ax_spec.set_title(f"Log-Mel Spectrogram Representation (Input Shape: 322 × 128 × 1)", color='#F8FAFC', fontsize=11, fontweight='600', pad=10)
        ax_spec.set_xlabel("Time (seconds)", color='#94A3B8', fontsize=9)
        ax_spec.set_ylabel("Mel Frequency Bands (0 - 128)", color='#94A3B8', fontsize=9)
        
        cbar = fig_spec.colorbar(im, ax=ax_spec, pad=0.015)
        cbar.set_label("Log Magnitude (dB)", color='#94A3B8', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='#94A3B8')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94A3B8', fontsize=7)
        cbar.outline.set_edgecolor('#1E293B')
        
        for spine in ax_spec.spines.values():
            spine.set_color('#1E293B')
        ax_spec.tick_params(colors='#94A3B8', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="gradient-title">🎙️ Speech Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Deep Learning Acoustic Emotion Classifier powered by a 2D CNN trained on Log-Mel Spectrograms (CREMA-D)</div>', unsafe_allow_html=True)

# Layout: Two columns (Left: Audio Input & Control; Right: Classification Results)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🎙️ Audio Input")
    
    input_tab_upload, input_tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])
    
    # ------------------ Tab 1: Upload Audio ------------------
    with input_tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a speech recording (WAV or MP3)",
            type=["wav", "mp3"],
            help="Upload an expressive speech sample. Max recommended duration: 10s.",
            key="file_uploader"
        )
        if uploaded_file is not None:
            up_bytes = uploaded_file.getvalue()
            up_ext = uploaded_file.name.split(".")[-1].lower()
            try:
                up_y, up_sr, up_dur, up_std, up_mel, up_feat = process_audio_file(up_bytes)
                st.markdown("#### 🔊 Audio Playback")
                st.audio(up_bytes, format=f"audio/{up_ext}")
                
                # Metadata stats
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{up_dur:.2f}s</div><div class="metric-label">Duration</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{up_sr:,} Hz</div><div class="metric-label">Input Rate</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">16,000 Hz</div><div class="metric-label">Standard Rate</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Predict Emotion", key="predict_upload_btn", use_container_width=True):
                    feat_batch = tf.expand_dims(up_feat, axis=0)
                    probs = model.predict(feat_batch, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    st.session_state["active_result"] = {
                        "pred_emotion": EMOTION_CLASSES[pred_idx],
                        "confidence": float(probs[pred_idx]),
                        "probs": probs,
                        "source_name": uploaded_file.name,
                        "audio_std": up_std,
                        "log_mel": up_mel
                    }
            except Exception as e:
                st.error(f"Error processing uploaded audio: {e}. Please ensure it is a valid, uncorrupted WAV or MP3 audio file.")

    # ------------------ Tab 2: Record Audio ------------------
    with input_tab_record:
        st.markdown(
            "<p style='color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.75rem;'>Click the microphone icon below to start recording, speak with emotional expression, and click stop when finished.</p>",
            unsafe_allow_html=True
        )
        recorded_audio = st.audio_input(
            "Record speech via microphone",
            help="Browser microphone access required. Click microphone to start/stop.",
            key="mic_recorder"
        )
        if recorded_audio is not None:
            rec_bytes = recorded_audio.getvalue()
            try:
                rec_y, rec_sr, rec_dur, rec_std, rec_mel, rec_feat = process_audio_file(rec_bytes)
                st.markdown("#### 🔊 Audio Playback")
                st.audio(rec_bytes, format="audio/wav")
                
                # Metadata stats
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{rec_dur:.2f}s</div><div class="metric-label">Duration</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{rec_sr:,} Hz</div><div class="metric-label">Input Rate</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">16,000 Hz</div><div class="metric-label">Standard Rate</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Predict Emotion", key="predict_record_btn", use_container_width=True):
                    feat_batch = tf.expand_dims(rec_feat, axis=0)
                    probs = model.predict(feat_batch, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    st.session_state["active_result"] = {
                        "pred_emotion": EMOTION_CLASSES[pred_idx],
                        "confidence": float(probs[pred_idx]),
                        "probs": probs,
                        "source_name": "Microphone Recording",
                        "audio_std": rec_std,
                        "log_mel": rec_mel
                    }
            except Exception as e:
                st.error(f"Error processing microphone recording: {e}. Please ensure the microphone was captured properly.")

with col_right:
    st.markdown("### 🎯 Classification Results")
    
    if "active_result" in st.session_state and st.session_state["active_result"] is not None:
        res = st.session_state["active_result"]
        render_prediction_card(res["pred_emotion"], res["confidence"], res["source_name"])
        st.markdown("#### 📊 Model Probability Distribution")
        render_probability_chart(res["probs"], res["pred_emotion"])
    elif uploaded_file is not None or ('recorded_audio' in locals() and recorded_audio is not None):
        st.info("Click **🔍 Predict Emotion** on the left to analyze the audio sample.")
    else:
        st.info("👈 Upload an audio file or record via microphone on the left, then click **Predict Emotion** to view acoustic classifications.")

# ---------------------------------------------------------
# Audio Visualization Area (Waveform & Spectrogram)
# ---------------------------------------------------------
if "active_result" in st.session_state and st.session_state["active_result"] is not None:
    res = st.session_state["active_result"]
    render_acoustic_visualizations(res["audio_std"], res["log_mel"], res["source_name"])

# ---------------------------------------------------------
# Footer / Academic Disclaimer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 1rem 0;">
    <b>Speech Emotion Recognition (SER) Project</b> &nbsp;|&nbsp; 
    Model: <code>models/baseline_cnn_best.keras</code> (28,294 parameters) &nbsp;|&nbsp;
    Architecture: 2D CNN (3 Conv Blocks + GlobalAveragePooling2D + Softmax) &nbsp;|&nbsp;
    Dataset: CREMA-D (Speaker-Independent)<br>
    <i>Note: Predictions represent acoustic prosody classifications (pitch, tempo, energy) rather than an objective psychological measurement of true internal emotion.</i>
</div>
""", unsafe_allow_html=True)
