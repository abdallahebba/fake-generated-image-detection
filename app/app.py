
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

CNN_PATH = MODEL_DIR / "CustomCNN.keras"
VGG_PATH = MODEL_DIR / "VGG16_transfer_best.keras"
RESNET_PATH = MODEL_DIR / "ResNet50_transfer_best.keras"

# Same ensemble weights used in the notebook
CNN_WEIGHT = 0.20
VGG_WEIGHT = 0.30
RESNET_WEIGHT = 0.50


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1.2rem;
        }

        /* Hero */
        .hero {
            padding: 1.7rem 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #111827 0%, #172554 55%, #312e81 100%);
            border: 1px solid #26345f;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: #cbd5e1;
            font-size: 1.02rem;
        }

        /* Primary button */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            border: 0;
            color: white;
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.25);
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8, #6d28d9);
            color: white;
            border: 0;
        }

        /* Upload box */
        [data-testid="stFileUploader"] {
            border-radius: 14px;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #64748b;
            border-radius: 14px;
            background: rgba(30, 41, 59, 0.35);
        }

        /* Result cards */
        .result-box {
            padding: 1.45rem;
            border-radius: 18px;
            text-align: center;
            margin: 0.85rem 0 1rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.10);
        }

        .real-box {
            background: linear-gradient(135deg, #052e2b, #064e3b);
            border: 1px solid #10b981;
            color: #ecfdf5;
        }

        .fake-box {
            background: linear-gradient(135deg, #450a0a, #7f1d1d);
            border: 1px solid #f87171;
            color: #fff1f2;
        }

        .result-label {
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0;
            color: inherit !important;
        }

        .confidence {
            font-size: 1.1rem;
            margin-top: 0.45rem;
            color: inherit !important;
        }

        .confidence strong {
            color: inherit !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 0.85rem;
        }

        /* Section headings */
        h2, h3 {
            letter-spacing: -0.01em;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    missing = [
        str(path)
        for path in [CNN_PATH, VGG_PATH, RESNET_PATH]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model file(s):\n" + "\n".join(missing)
        )

    cnn = tf.keras.models.load_model(CNN_PATH, compile=False)
    vgg = tf.keras.models.load_model(VGG_PATH, compile=False)
    resnet = tf.keras.models.load_model(RESNET_PATH, compile=False)

    return cnn, vgg, resnet


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return array


# ============================================================
# Prediction
# ============================================================

def predict(image, cnn, vgg, resnet):
    x = preprocess_image(image)

    cnn_prob = float(cnn.predict(x, verbose=0)[0][0])
    vgg_prob = float(vgg.predict(x, verbose=0)[0][0])
    resnet_prob = float(resnet.predict(x, verbose=0)[0][0])

    ensemble_prob = (
        CNN_WEIGHT * cnn_prob
        + VGG_WEIGHT * vgg_prob
        + RESNET_WEIGHT * resnet_prob
    )

    return cnn_prob, vgg_prob, resnet_prob, ensemble_prob


def result_from_probability(probability):
    if probability >= 0.5:
        return "FAKE", probability * 100.0

    return "REAL", (1.0 - probability) * 100.0


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔍 AI Image Detector</h1>
        <p>
            Detect whether an image is real or AI-generated using
            three deep learning models and a weighted ensemble.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load Models
# ============================================================

try:
    cnn_model, vgg_model, resnet_model = load_models()
except Exception as error:
    st.error("Could not load the AI models.")
    st.code(str(error))
    st.info(
        "Make sure the three .keras files are inside the project's "
        "'models' folder."
    )
    st.stop()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Model Setup")

    st.write("**Ensemble weights**")

    st.write(f"Custom CNN: **{CNN_WEIGHT:.0%}**")
    st.write(f"VGG16: **{VGG_WEIGHT:.0%}**")
    st.write(f"ResNet50: **{RESNET_WEIGHT:.0%}**")

    st.divider()

    st.caption("Input size: 224 × 224")
    st.caption("Three-model weighted ensemble")
    st.caption("Inference only — no training")


# ============================================================
# Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP",
)


if uploaded_file is None:
    st.info("👆 Upload an image above to start the analysis.")
    st.stop()


image = Image.open(uploaded_file).convert("RGB")


# ============================================================
# Image Preview + Analyze
# ============================================================

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)
    st.caption(
        f"{image.width} × {image.height} pixels • {uploaded_file.name}"
    )

with right:
    st.subheader("Detection")

    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        st.info("Click **Analyze Image** to run all three models.")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            (
                cnn_prob,
                vgg_prob,
                resnet_prob,
                ensemble_prob,
            ) = predict(
                image,
                cnn_model,
                vgg_model,
                resnet_model,
            )
        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))
            st.stop()

    cnn_label, cnn_conf = result_from_probability(cnn_prob)
    vgg_label, vgg_conf = result_from_probability(vgg_prob)
    resnet_label, resnet_conf = result_from_probability(resnet_prob)
    ensemble_label, ensemble_conf = result_from_probability(ensemble_prob)

    # --------------------------------------------------------
    # Final Ensemble Result
    # --------------------------------------------------------

    st.subheader("🏆 Final Result")

    if ensemble_label == "REAL":
        st.markdown(
            f"""
            <div class="result-box real-box">
                <p class="result-label">✅ REAL</p>
                <p class="confidence">
                    Confidence: <strong>{ensemble_conf:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-box fake-box">
                <p class="result-label">⚠️ FAKE / AI-GENERATED</p>
                <p class="confidence">
                    Confidence: <strong>{ensemble_conf:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Model Results
# ============================================================

st.divider()
st.subheader("Individual Model Predictions")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Custom CNN",
        cnn_label,
        f"{cnn_conf:.2f}% confidence",
    )

with col2:
    st.metric(
        "VGG16",
        vgg_label,
        f"{vgg_conf:.2f}% confidence",
    )

with col3:
    st.metric(
        "ResNet50",
        resnet_label,
        f"{resnet_conf:.2f}% confidence",
    )


# ============================================================
# Ensemble Details
# ============================================================

st.divider()
st.subheader("Ensemble Decision")

ensemble_col1, ensemble_col2 = st.columns([1, 1])

with ensemble_col1:
    st.write("**Model contribution**")
    st.write(f"Custom CNN — {CNN_WEIGHT:.0%}")
    st.progress(CNN_WEIGHT)

    st.write(f"VGG16 — {VGG_WEIGHT:.0%}")
    st.progress(VGG_WEIGHT)

    st.write(f"ResNet50 — {RESNET_WEIGHT:.0%}")
    st.progress(RESNET_WEIGHT)

with ensemble_col2:
    st.write("**Final ensemble confidence**")
    st.progress(min(int(round(ensemble_conf)), 100))

    st.write(f"Prediction: **{ensemble_label}**")
    st.write(f"Confidence: **{ensemble_conf:.2f}%**")

    st.caption(
        "The final decision is calculated from the weighted "
        "FAKE probabilities of the three models."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "AI Image Detector • Custom CNN + VGG16 + ResNet50 • "
    "Weighted Ensemble"
)
