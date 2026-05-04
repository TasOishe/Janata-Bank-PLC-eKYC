import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="NID Info Extractor",
    page_icon="🪪",
    layout="centered",
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Bengali:wght@400;600&family=Sora:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { padding: 2rem 1.5rem; }

h1 { font-size: 2rem !important; font-weight: 700 !important; color: #58d1c2 !important; letter-spacing: -0.5px; }
h3 { color: #8b949e !important; font-weight: 400 !important; margin-top: -0.5rem !important; }

section[data-testid="stFileUploader"],
div[data-testid="stCameraInput"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
}

.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
}

.result-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.65rem 0;
    border-bottom: 1px solid #21262d;
    gap: 1rem;
}
.result-row:last-child { border-bottom: none; }

.result-label {
    color: #58d1c2;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    min-width: 130px;
    padding-top: 2px;
}

.result-value {
    color: #e6edf3;
    font-size: 1rem;
    font-family: 'IBM Plex Sans Bengali', 'Sora', sans-serif;
    text-align: right;
    flex: 1;
}

.not-found { color: #484f58 !important; font-style: italic; font-size: 0.9rem !important; }

.badge {
    display: inline-block;
    background: #1f3d39;
    color: #58d1c2;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 8px;
    border-radius: 999px;
    margin-bottom: 1rem;
    letter-spacing: 0.6px;
}

.stButton > button {
    background: #58d1c2 !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.6rem !important;
    font-family: 'Sora', sans-serif !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    color: #58d1c2 !important;
    border-bottom: 2px solid #58d1c2 !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Load models 
# ────────────────----------------------------------------------------─────────────────────────────
@st.cache_resource(show_spinner="Loading OCR engine…")
def load_reader():
    return easyocr.Reader(['en', 'bn'])

@st.cache_resource(show_spinner="Loading face detector…")
def load_detector():
    # Using OpenCV's built-in Haar cascade 
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)


# ----------------------------------------------------
# Core processing
# ----------------------------------------------------
def crop_nid_region(img_array, detector):
    img  = cv2.resize(img_array, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        st.warning("No face detected — running OCR on the full image.")
        return gray

    x, y, w, h = faces[0]
    x2, y1, y2 = x + w, y, y + h

    cropped = img[y1 - 80 : y2 + 170, x2 + 10 : x2 + 420]

    if cropped.size == 0:
        st.warning("Crop region was empty — using full image.")
        return gray

    # gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # denoised  = cv2.fastNlMeansDenoising(gray_crop, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced  = clahe.apply(denoised)
    # return enhanced

    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray_crop, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE for contrast
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Morphological opening removes thin wavy background
    #while keeping thick text strokes intact
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
 
    # Upscale 2x before returning, improves OCR on small  text
    cleaned = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cleaned



def extract_nid_info(img_array, reader):
    results   = reader.readtext(img_array)
    texts     = [res[1] for res in results]
    full_text = " ".join(texts)

    info = {
        "Name (EN)"    : "Not Found",
        "Name (BN)"    : "Not Found",
        "Father"       : "Not Found",
        "Mother"       : "Not Found",
        "Date of Birth": "Not Found",
        "NID No."      : "Not Found",
    }

    # NID number 
    for match in re.finditer(r'\b\d[\d\s-]*\d\b', full_text):
        cleaned = re.sub(r'\D', '', match.group(0))
        if len(cleaned) == 10:
            info["NID No."] = cleaned
            break

    # Date of birth
    dob = re.search(
        r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4}\b',
        full_text
    )
    if dob:
        info["Date of Birth"] = dob.group()

    # English name (all-caps)
    # for text in texts:
    #     if text.isupper() and len(text) > 5:
    #         info["Name (EN)"] = text
    #         break

    # English name (all-caps, letters only)
    for text in texts:
        if text.isupper() and len(text) > 5 and re.match(r'^[A-Z\s\.\-]+$', text):
            info["Name (EN)"] = text
            break

    # Bangla fields
    # from thefuzz import fuzz

    # def is_keyword_match(text, keyword, threshold=50):
    #     # Only consider tokens that are similar in length to the keyword
    #     if abs(len(text) - len(keyword)) > 3:
    #         return False
    #     return fuzz.ratio(keyword, text) >= threshold

    # name_found = False
    # for i, text in enumerate(texts):
    #     if is_keyword_match(text, "নাম") and i + 1 < len(texts):
    #         info["Name (BN)"] = texts[i + 1]
    #         name_found = True
    #     if is_keyword_match(text, "পিতা") and i + 1 < len(texts):
    #         info["Father"] = texts[i + 1]
    #     if is_keyword_match(text, "মাতা") and i + 1 < len(texts):
    #         info["Mother"] = texts[i + 1]

    # if not name_found:
    #     for text in texts:
    #         if re.search(r'[\u0980-\u09FF]', text) and len(text) >= 5:     
    #             info["Name (BN)"] = text
    #             break
    # Bangla fields
    from thefuzz import fuzz

    def is_keyword_match(text, keyword, threshold=60):
        if abs(len(text) - len(keyword)) > 3:
            return False
        return fuzz.ratio(keyword, text) >= threshold

    # Collect all bangla tokens in order (excluding short noise)
    bangla_tokens = [t for t in texts if re.search(r'[\u0980-\u09FF]', t) and len(t) >= 5]

    name_found   = False
    father_found = False
    mother_found = False

    for i, text in enumerate(texts):
        if is_keyword_match(text, "নাম") and i + 1 < len(texts):
            info["Name (BN)"] = texts[i + 1]
            name_found = True
        if is_keyword_match(text, "পিতা") and i + 1 < len(texts):
            info["Father"] = texts[i + 1]
            father_found = True
        if is_keyword_match(text, "মাতা") and i + 1 < len(texts):
            info["Mother"] = texts[i + 1]
            mother_found = True

    # Fallbacks using bangla token order
    # Order on NID is always: Name → Father → Mother
    if not name_found and len(bangla_tokens) >= 1:
        info["Name (BN)"] = bangla_tokens[0]

    if not father_found and len(bangla_tokens) >= 2:
        info["Father"] = bangla_tokens[1]

    if not mother_found and len(bangla_tokens) >= 3:
        info["Mother"] = bangla_tokens[2]

    return info, texts


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.markdown("# 🪪 NID Info Extractor")
st.markdown("### Extract key fields from a Bangladeshi National ID card")
st.markdown("---")

reader   = load_reader()
detector = load_detector()

image_input = None

uploaded = st.file_uploader(
    "Choose an NID image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
if uploaded:
    image_input = Image.open(uploaded)
    st.image(image_input, caption="Uploaded Image", use_container_width=True)

if image_input:
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run = st.button("🔍  Extract Info", use_container_width=True)

    if run:
        with st.spinner("Processing image…"):
            cv_img      = pil_to_cv2(image_input)
            cropped     = crop_nid_region(cv_img, detector)
            info, texts = extract_nid_info(cropped, reader)

        # Results card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge">EXTRACTED FIELDS</span>', unsafe_allow_html=True)

        for label, value in info.items():
            val_class = "not-found" if value == "Not Found" else ""
            st.markdown(f"""
            <div class="result-row">
                <span class="result-label">{label}</span>
                <span class="result-value {val_class}">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Raw OCR tokens"):
            st.write(texts)

        with st.expander("Preprocessed region used for OCR"):
            st.image(cropped, clamp=True, use_container_width=True)

else:
    st.info("Upload an NID image or use your camera to get started.", icon="ℹ️")








