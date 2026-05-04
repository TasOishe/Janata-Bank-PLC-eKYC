# 🪪 e-KYC — NID Information Extractor
### Phase 1: Automated Customer Identification from National ID Cards

> Part of the **e-KYC (Electronically Know Your Customer)** project.  
> This phase detects and extracts customer information directly from a photo of their Bangladeshi National ID (NID) card — no manual data entry required.

---

## 📌 Overview

Traditional KYC processes require customers to manually fill in their information, which is slow, error-prone, and inconvenient. This project automates Phase 1 of e-KYC by using computer vision and OCR (Optical Character Recognition) to extract key fields from an NID card image automatically.

A Streamlit web app is included as a demo interface — it accepts an NID image, processes it through the pipeline, and displays the extracted fields along with the raw OCR output and the preprocessed image region used for detection.

---

## 🧩 Extracted Fields

| Field | Description |
|---|---|
| Name (English) | Full name in English (uppercase) |
| Name (Bangla) | Full name in Bangla script |
| Father's Name | Father's name in Bangla |
| Mother's Name | Mother's name in Bangla |
| Date of Birth | In `DD Mon YYYY` format |
| NID Number | 10-digit National ID number |

---

## 🔄 How It Works

```
Input Image (NID photo)
        │
        ▼
┌───────────────────────┐
│   Face Detection      │  ← OpenCV Haar Cascade detects the face
│   (Haar Cascade)      │    on the NID card
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Region Cropping     │  ← Crops the info region to the right
│                       │    of the detected face
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Image Enhancement   │  ← Denoising → CLAHE contrast boost
│   Pipeline            │    → Morphological cleaning
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   OCR                 │  ← EasyOCR reads both English
│   (EasyOCR)           │    and Bangla text
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Field Parsing       │  ← Regex + fuzzy keyword matching
│                       │    extracts structured fields
└───────────┬───────────┘
            │
            ▼
  Structured Output
  (Name, DOB, NID No., etc.)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web App | Streamlit |
| Face Detection | OpenCV Haar Cascade |
| OCR Engine | EasyOCR (English + Bangla) |
| Image Processing | OpenCV, NumPy |
| Keyword Matching | thefuzz (fuzzy string matching) |
| Language | Python 3.10 |

---

## 📁 Project Structure

```
nid_app/
│
├── nid_extractor_app.py   # Main Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.10** — download from [python.org/downloads/release/python-31011](https://python.org/downloads/release/python-31011)
  - ⚠️ During install, check **"Add Python to PATH"**
- Python 3.11+ and 3.14 are **not supported** — EasyOCR and dlib wheels are unavailable for those versions

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

### Step 2 — Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` — this means the environment is active.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages. EasyOCR will also download its language model files on first run (~300MB), so ensure you have a stable internet connection.

---

### Step 4 — Run the app

```bash
streamlit run nid_extractor_app.py
```

Your browser will automatically open at `http://localhost:8501`.

---

## 📱 Running on Mobile (Same Wi-Fi Network)

You can access the app from your phone as long as both devices are on the same Wi-Fi.

**Step 1 — Find your PC's local IP:**
```bash
ipconfig        # Windows
ifconfig        # Mac/Linux
```
Look for the **IPv4 Address** under your Wi-Fi adapter, e.g. `192.168.1.105`

**Step 2 — Run the app:**
```bash
streamlit run nid_extractor_app.py
```

**Step 3 — Open on your phone:**
```
http://192.168.1.105:8501
```

> **Note:** The phone's browser may block camera access on plain `http://`. To get HTTPS (required for camera), use [ngrok](https://ngrok.com):
> ```bash
> ngrok http 8501
> ```
> Then open the `https://...` URL ngrok provides on your phone.

---

## 🔍 Pipeline Details

### 1. Face Detection & Cropping

OpenCV's Haar Cascade classifier detects the face on the NID card. The information region (name, father, mother, DOB, NID number) is always to the right of the photo on Bangladeshi NID cards, so the pipeline crops a fixed region relative to the detected face bounding box.

If no face is detected, the full image is passed to OCR as a fallback.

### 2. Image Enhancement

The cropped region goes through a three-step enhancement pipeline:

| Step | Method | Purpose |
|---|---|---|
| Denoising | `fastNlMeansDenoising` (h=10) | Removes camera noise without blurring text |
| Contrast boost | CLAHE (clipLimit=2.0) | Improves local contrast for faint text |
| Background suppression | Morphological closing | Suppresses the NID card's wavy security pattern |

### 3. OCR

EasyOCR is used with both `en` (English) and `bn` (Bangla) language models. Key parameters tuned for NID cards:

```python
reader.readtext(
    img_array,
    text_threshold=0.6,   # accept lower-confidence Bangla detections
    low_text=0.3,         # pick up faint matras and characters
    link_threshold=0.3,   # keep Bangla characters properly linked
    width_ths=1.0,        # prevent words from being split
)
```

### 4. Field Parsing

**English Name** — detected as the first all-uppercase token that contains only letters, spaces, dots and hyphens (filters out garbage tokens).

**NID Number** — regex finds sequences of digits and strips non-digits, accepting only 10-digit results.

**Date of Birth** — regex matches the pattern `DD Mon YYYY` (e.g. `21 Sep 1985`).

**Bangla Fields** — uses fuzzy keyword matching (`thefuzz`) to find নাম, পিতা, মাতা even when OCR misreads the keyword slightly (e.g. পিতা read as গীতা). If a keyword is missed entirely, the system falls back to positional detection — since Bangladeshi NID cards always follow the same layout order (Name → Father → Mother), the Bangla tokens are picked by position.

---

## 🖥️ Demo App Features

| Feature | Description |
|---|---|
| Image upload | Accepts JPG, PNG, JPEG |
| Extracted fields card | Displays all 6 fields with clear labels |
| Raw OCR tokens | Expandable section showing every token EasyOCR detected |
| Preprocessed image | Expandable section showing the cropped and enhanced region that was sent to OCR — useful for debugging |

---

## ⚠️ Known Limitations

- **Face must be visible** — if the face on the NID card is obscured or cut off, the crop falls back to the full image which may reduce accuracy
- **Image quality matters** — very blurry or low-light photos will reduce OCR accuracy; good lighting and a steady hand give the best results
- **Bangla keyword OCR** — small keyword labels like পিতা and মাতা are sometimes misread; the fuzzy matching and positional fallback handle most cases but not all
- **10-digit NID only** — older 13 or 17-digit NID formats are not currently handled

---

---

## 📦 Dependencies

```
streamlit>=1.35.0
easyocr>=1.7.1
opencv-python-headless>=4.9.0
numpy>=1.26.0
Pillow>=10.0.0
thefuzz>=0.22.1
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

Developed as part of an internship e-KYC project.
