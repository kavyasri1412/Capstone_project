import os
import json
import re
import hashlib

import streamlit as st
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ollama import chat


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Intelligent Resume & Job Matching Platform"

# Always use the same folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RESUME_DIR = os.path.join(
    DATA_DIR,
    "resumes"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    RESUME_DIR,
    exist_ok=True
)

os.makedirs(
    CHROMA_DIR,
    exist_ok=True
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background: #f4f7fb;
}
/* ===== MAKE MAIN CONTENT TEXT BLACK ===== */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp li,
.stApp td,
.stApp th,
.stApp input,
.stApp textarea {
    color: #000000;
}

/* Keep sidebar text white */
section[data-testid="stSidebar"] * {
    color: #f5f5f5 !important;
}

/* ============================================================
   DASHBOARD HERO - FORCE ALL TEXT TO WHITE
   ============================================================ */

.dashboard-hero {
    color: #ffffff !important;
}

.dashboard-hero *,
.dashboard-hero p,
.dashboard-hero span,
.dashboard-hero div,
.dashboard-hero h1,
.dashboard-hero h2,
.dashboard-hero h3 {
    color: #ffffff !important;
}

/* Hero title */
.hero-title {
    color: #ffffff !important;
}

/* Highlighted part - WHITE, NOT BLUE/CYAN */
.hero-highlight {
    color: #ffffff !important;
    background: none !important;
    -webkit-background-clip: initial !important;
    -webkit-text-fill-color: #ffffff !important;
    background-clip: initial !important;
}

/* Small top label */
.hero-eyebrow {
    color: #ffffff !important;
}

/* Description */
.hero-description {
    color: rgba(255,255,255,0.88) !important;
}

/* Feature pills */
.hero-pill {
    color: #ffffff !important;
}

/* Keep profile card text white */
.profile-card,
.profile-card * {
    color: white;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111111 0%,
        #181818 55%,
        #202020 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #f5f5f5 !important;
}

/* ===== PROFESSIONAL BLACK & WHITE SIDEBAR NAVIGATION ===== */

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 6px !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin: 3px 0 !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

/* Remove the default radio dot */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Normal navigation text */
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #d4d4d4 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Hover */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #262626 !important;
    border-color: #333333 !important;
}

/* Selected navigation item */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #f5f5f5 !important;
    border-color: #f5f5f5 !important;
    box-shadow: none !important;
}

/* Selected navigation text */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #111111 !important;
    font-weight: 700 !important;
}

.sidebar-brand {
    padding: 14px 8px 24px 8px;
}

.sidebar-brand-title {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.sidebar-brand-subtitle {
    font-size: 11px;
    color: #aebbd0 !important;
    margin-top: 5px;
    line-height: 1.5;
}

/* BUTTONS */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #dbe3ef;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #6366f1;
    box-shadow: 0 6px 18px rgba(79,70,229,0.10);
}

/* DASHBOARD HERO */
.dashboard-hero {
    position: relative;
    overflow: hidden;
    min-height: 300px;
    padding: 38px 42px;
    border-radius: 28px;
    margin-bottom: 22px;
    color: white;
    background:
        radial-gradient(circle at 87% 18%, rgba(129,140,248,0.38), transparent 24%),
        radial-gradient(circle at 72% 90%, rgba(34,211,238,0.16), transparent 26%),
        linear-gradient(135deg, #07111f 0%, #10204a 52%, #312e81 100%);
    box-shadow: 0 24px 55px rgba(15,23,42,0.18);
}

.dashboard-hero::before,
.dashboard-hero::after {
    content: "";
    position: absolute;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 50%;
    pointer-events: none;
}

.dashboard-hero::before {
    width: 270px;
    height: 270px;
    right: -75px;
    top: -110px;
}

.dashboard-hero::after {
    width: 190px;
    height: 190px;
    right: 90px;
    bottom: -125px;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 850px;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 13px;
    border-radius: 30px;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.13);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.1px;
    color: #dbeafe;
    margin-bottom: 17px;
}

.hero-title {
    font-size: 40px;
    line-height: 1.08;
    font-weight: 850;
    letter-spacing: -1.5px;
    margin: 0;
}

.hero-highlight {
    background: linear-gradient(90deg, #c7d2fe, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-description {
    margin-top: 15px;
    max-width: 700px;
    font-size: 14px;
    line-height: 1.75;
    color: rgba(255,255,255,0.74);
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 22px;
}

.hero-pill {
    padding: 7px 11px;
    border-radius: 9px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 650;
}

/* SECTION HEADINGS */
.dashboard-section-title {
    margin-top: 26px;
    margin-bottom: 4px;
    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.35px;
}

.dashboard-section-subtitle {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 15px;
}

/* KPI CARDS */
.kpi-card {
    position: relative;
    overflow: hidden;
    min-height: 126px;
    padding: 20px;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid #e6ebf2;
    box-shadow: 0 7px 24px rgba(15,23,42,0.055);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 30px rgba(15,23,42,0.09);
}

.kpi-card::after {
    content: "";
    position: absolute;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    right: -28px;
    top: -28px;
    background: #eef2ff;
}

.kpi-icon {
    position: relative;
    z-index: 1;
    font-size: 21px;
}

.kpi-label {
    margin-top: 11px;
    color: #64748b;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.kpi-value {
    margin-top: 2px;
    color: #0f172a;
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -0.6px;
}

.kpi-caption {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 1px;
}

/* INTELLIGENCE PANEL */
.intelligence-card {
    min-height: 250px;
    padding: 27px;
    border-radius: 21px;
    background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 27px rgba(15,23,42,0.055);
}

.intelligence-label {
    color: #4f46e5;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1px;
}

.intelligence-title {
    margin-top: 10px;
    color: #111827;
    font-size: 22px;
    font-weight: 820;
    letter-spacing: -0.4px;
}

.intelligence-text {
    margin-top: 9px;
    color: #475569;
    font-size: 13px;
    line-height: 1.7;
}

.insight-box {
    margin-top: 18px;
    padding: 13px 15px;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    color: #334155;
    font-size: 12px;
    line-height: 1.55;
}

/* PROFILE STATUS */
.profile-card {
    min-height: 250px;
    padding: 27px;
    border-radius: 21px;
    background: #0b1220;
    color: white;
    box-shadow: 0 12px 30px rgba(15,23,42,0.14);
    position: relative;
    overflow: hidden;
}

.profile-card::before {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    right: -75px;
    top: -75px;
    border-radius: 50%;
    border: 24px solid rgba(129,140,248,0.10);
}

.profile-label {
    position: relative;
    color: #a5b4fc;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1px;
}

.profile-title {
    position: relative;
    margin-top: 9px;
    font-size: 23px;
    font-weight: 820;
}

.profile-description {
    position: relative;
    margin-top: 9px;
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.6;
}

.profile-progress {
    position: relative;
    height: 8px;
    margin-top: 20px;
    border-radius: 20px;
    background: #263244;
    overflow: hidden;
}

.profile-progress-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}

.profile-progress-text {
    position: relative;
    margin-top: 8px;
    color: #94a3b8;
    font-size: 10px;
}

/* SKILLS */
.skill-panel {
    padding: 23px;
    border-radius: 19px;
    background: white;
    border: 1px solid #e6ebf2;
    box-shadow: 0 7px 23px rgba(15,23,42,0.05);
    min-height: 205px;
}

.skill-chip {
    display: inline-block;
    padding: 8px 11px;
    margin: 5px 5px 0 0;
    border-radius: 9px;
    background: #f1f5ff;
    border: 1px solid #dbe4ff;
    color: #3730a3;
    font-size: 11px;
    font-weight: 700;
}

.empty-skill {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.6;
    padding-top: 10px;
}

/* ACTION PANEL */
.action-panel {
    padding: 23px;
    border-radius: 19px;
    background: white;
    border: 1px solid #e6ebf2;
    box-shadow: 0 7px 23px rgba(15,23,42,0.05);
    min-height: 205px;
}

.action-item {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    padding: 10px 0;
    border-bottom: 1px solid #eef2f7;
}

.action-item:last-child {
    border-bottom: 0;
}

.action-icon {
    width: 29px;
    height: 29px;
    flex: 0 0 29px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    background: #eef2ff;
    font-size: 14px;
}

.action-name {
    color: #1e293b;
    font-size: 12px;
    font-weight: 750;
}

.action-desc {
    margin-top: 2px;
    color: #94a3b8;
    font-size: 10px;
    line-height: 1.45;
}

/* JOURNEY */
.journey-panel {
    padding: 24px 18px;
    border-radius: 20px;
    background: white;
    border: 1px solid #e6ebf2;
    box-shadow: 0 7px 23px rgba(15,23,42,0.05);
}

.journey-wrap {
    display: flex;
    align-items: flex-start;
    width: 100%;
}

.journey-step {
    flex: 1;
    text-align: center;
}

.journey-circle {
    width: 40px;
    height: 40px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #eef2ff;
    border: 2px solid #c7d2fe;
    color: #4338ca;
    font-size: 13px;
    font-weight: 850;
}

.journey-circle.done {
    background: #e0e7ff;
    border-color: #818cf8;
}

.journey-name {
    margin-top: 8px;
    color: #334155;
    font-size: 10px;
    font-weight: 750;
}

.journey-line {
    flex: 0.45;
    height: 2px;
    margin-top: 20px;
    background: #e2e8f0;
}

/* TOOLKIT */
.tool-card {
    min-height: 145px;
    padding: 19px;
    border-radius: 17px;
    background: white;
    border: 1px solid #e6ebf2;
    box-shadow: 0 6px 20px rgba(15,23,42,0.045);
    transition: all 0.2s ease;
}

.tool-card:hover {
    transform: translateY(-3px);
    border-color: #c7d2fe;
    box-shadow: 0 12px 25px rgba(15,23,42,0.08);
}

.tool-icon {
    font-size: 22px;
}

.tool-title {
    margin-top: 10px;
    color: #1e293b;
    font-size: 13px;
    font-weight: 800;
}

.tool-description {
    margin-top: 4px;
    color: #64748b;
    font-size: 10px;
    line-height: 1.5;
}

.dashboard-footer {
    text-align: center;
    padding: 30px 10px 10px;
    color: #94a3b8;
    font-size: 11px;
}

/* DATAFRAME */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ===== HEADING COLOR FIX ===== */

.section-title,
.section-heading,
.card-title,
.dashboard-heading {
    color: #172033 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #172033 !important;
}

/* HERO HEADINGS MUST STAY WHITE */
.dashboard-hero h1,
.dashboard-hero h2,
.dashboard-hero h3,
.dashboard-hero h4,
.dashboard-hero h5,
.dashboard-hero h6 {
    color: #ffffff !important;
}

/* ===== RESUME ANALYZER TEXT COLOR ===== */

div[data-testid="stFileUploader"] {
    color: #000000 !important;
}

div[data-testid="stFileUploader"] * {
    color: #000000 !important;
}

div[data-testid="stFileUploader"] button {
    color: #000000 !important;
}

div[data-testid="stFileUploader"] button * {
    color: #000000 !important;
}

.stButton > button {
    color: #000000 !important;
}

.stButton > button * {
    color: #000000 !important;
}

/* ===== WHITE ANALYSIS BOX ===== */

div[data-testid="stStatus"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 16px !important;
    color: #111111 !important;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08) !important;
}

div[data-testid="stStatus"] * {
    background-color: #ffffff !important;
    color: #111111 !important;
}

div[data-testid="stStatus"] [role="progressbar"] {
    background-color: #e5e7eb !important;
}

div[data-testid="stStatus"] [role="progressbar"] > div {
    background-color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "latest_analysis" not in st.session_state:
    st.session_state["latest_analysis"] = {}

if "job_matches" not in st.session_state:
    st.session_state["job_matches"] = []

if "career_recommendations" not in st.session_state:
    st.session_state["career_recommendations"] = ""

if "career_messages" not in st.session_state:
    st.session_state["career_messages"] = []


# ============================================================
# CACHED AI MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    embedding_model = load_embedding_model()
except Exception as e:
    embedding_model = None


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


client = get_chroma_client()


@st.cache_resource
def get_resume_collection():
    return client.get_or_create_collection(
        name="resume_documents"
    )


@st.cache_resource
def get_job_collection():
    return client.get_or_create_collection(
        name="job_documents"
    )


resume_collection = get_resume_collection()
job_collection = get_job_collection()


# ============================================================
# PDF FUNCTIONS
# ============================================================

def extract_pdf_text(file_path):

    text = ""

    try:
        reader = PdfReader(file_path)

        if not reader.pages:
            return ""

        for page in reader.pages:

            try:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            except Exception:
                continue

    except Exception as e:
        raise RuntimeError(
            f"Error reading PDF: {str(e)}"
        )

    return text.strip()


# ============================================================
# EMBEDDING
# ============================================================

@st.cache_data(show_spinner=False)
def create_embedding(text):

    if not text or not text.strip():
        return []

    if embedding_model is None:
        raise RuntimeError(
            "Sentence Transformer model could not be loaded."
        )

    embedding = embedding_model.encode(
        text[:12000],
        normalize_embeddings=True
    )

    return embedding.tolist()


# ============================================================
# OLLAMA
# ============================================================

# ============================================================
# OLLAMA
# ============================================================

def ask_ai(prompt, json_mode=False):

    try:

        chat_options = {
            "model": "llama3.2",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # JSON mode is used only for Resume Analysis
        if json_mode:
            chat_options["format"] = "json"

        response = chat(**chat_options)

        # Current Ollama Python response format
        try:
            content = response.message.content

            if content:
                return str(content).strip()

        except Exception:
            pass

        # Fallback for dictionary-style response
        try:
            content = response["message"]["content"]

            if content:
                return str(content).strip()

        except Exception:
            pass

        return ""

    except Exception as e:

        return (
            "__AI_ERROR__"
            + str(e)
        )
# ============================================================
# SAFE JSON EXTRACTION
# ============================================================

def extract_json(text):

    if not text:
        return {}

    text = str(text).strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    text = text.strip()

    # --------------------------------------------------------
    # 1. Try direct JSON
    # --------------------------------------------------------

    try:

        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Try to locate JSON object inside AI response
    # --------------------------------------------------------

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):

        start = match.start()

        try:

            data, _ = decoder.raw_decode(
                text[start:]
            )

            if isinstance(data, dict):
                return data

        except Exception:
            continue

    return {}

# ============================================================
# NORMALIZE ANALYSIS
# ============================================================

def normalize_analysis(analysis):

    if not isinstance(analysis, dict):
        analysis = {}

    def safe_list(value):

        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if isinstance(value, str) and value.strip():
            return [value.strip()]

        return []

    summary = analysis.get(
        "summary",
        "No summary was generated."
    )

    if not isinstance(summary, str):
        summary = str(summary)

    score = analysis.get(
        "match_score",
        0
    )

    try:
        score = float(score)
        score = max(0, min(100, score))

        if score.is_integer():
            score = int(score)

    except Exception:
        score = 0

    return {
        "summary": summary,
        "skills": safe_list(
            analysis.get("skills", [])
        ),
        "strengths": safe_list(
            analysis.get("strengths", [])
        ),
        "weaknesses": safe_list(
            analysis.get("weaknesses", [])
        ),
        "missing_skills": safe_list(
            analysis.get("missing_skills", [])
        ),
        "recommended_roles": safe_list(
            analysis.get("recommended_roles", [])
        ),
        "match_score": score
    }


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_records(record_type):

    collection = (
        resume_collection
        if record_type == "resume"
        else job_collection
    )

    try:

        result = collection.get(
            include=[
                "metadatas",
                "documents"
            ]
        )

        records = []

        ids = result.get(
            "ids",
            []
        )

        metadatas = result.get(
            "metadatas",
            []
        )

        documents = result.get(
            "documents",
            []
        )

        for i, record_id in enumerate(ids):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                and metadatas[i] is not None
                else {}
            )

            document = (
                documents[i]
                if i < len(documents)
                and documents[i] is not None
                else ""
            )

            records.append({
                "id": record_id,
                "metadata": metadata,
                "document": document
            })

        return records

    except Exception as e:

        return []


# ============================================================
# RESUME DATABASE
# ============================================================

def add_resume_to_database(
    file_name,
    display_name,
    text
):

    if not text or not text.strip():
        raise ValueError(
            "Resume text is empty."
        )

    # Include text in the ID so replacing a file
    # with the same name creates the correct record.
    raw_id = (
        file_name
        + "|"
        + hashlib.md5(
            text.encode(
                "utf-8",
                errors="ignore"
            )
        ).hexdigest()
    )

    record_id = hashlib.md5(
        raw_id.encode("utf-8")
    ).hexdigest()

    embedding = create_embedding(text)

    if not embedding:
        raise ValueError(
            "Could not create resume embedding."
        )

    resume_collection.upsert(
        ids=[record_id],
        documents=[text[:12000]],
        embeddings=[embedding],
        metadatas=[{
            "type": "resume",
            "name": display_name,
            "file_name": file_name
        }]
    )

    return record_id


def delete_resume(
    record_id,
    file_name
):

    try:

        resume_collection.delete(
            ids=[record_id]
        )

    except Exception:
        pass

    file_path = os.path.join(
        RESUME_DIR,
        file_name
    )

    if os.path.exists(file_path):

        try:
            os.remove(file_path)

        except Exception:
            pass


# ============================================================
# JOB DATABASE
# ============================================================

def add_job_to_database(
    title,
    company,
    description
):

    title = title.strip()
    company = company.strip()
    description = description.strip()

    if not title:
        raise ValueError(
            "Job title is required."
        )

    if not company:
        raise ValueError(
            "Company name is required."
        )

    if not description:
        raise ValueError(
            "Job description is required."
        )

    raw_id = (
        title
        + "|"
        + company
        + "|"
        + description
    )

    job_id = hashlib.md5(
        raw_id.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()

    embedding = create_embedding(
        description
    )

    if not embedding:
        raise ValueError(
            "Could not create job embedding."
        )

    job_collection.upsert(
        ids=[job_id],
        documents=[description[:12000]],
        embeddings=[embedding],
        metadatas=[{
            "type": "job",
            "title": title,
            "company": company
        }]
    )

    return job_id


def delete_job(record_id):

    try:

        job_collection.delete(
            ids=[record_id]
        )

    except Exception:
        pass


# ============================================================
# RESUME ANALYSIS
# ============================================================

def analyze_resume(resume_text):

    if not resume_text or not resume_text.strip():
        return {
            "error": "Resume text is empty."
        }

    prompt = f"""
You are an expert AI resume analyzer.

Analyze the resume given below.

You MUST return ONLY a valid JSON object.

Do NOT write:
- Markdown
- ```json
- explanations
- comments
- text before the JSON
- text after the JSON

Use EXACTLY these keys:

{{
    "summary": "A short professional summary of the candidate.",
    "skills": ["Python", "Java", "SQL"],
    "strengths": ["Strong programming skills", "Good project experience"],
    "weaknesses": ["Limited industry experience"],
    "missing_skills": ["Advanced SQL", "Cloud deployment"],
    "recommended_roles": ["Data Analyst", "Python Developer"],
    "match_score": 75
}}

IMPORTANT RULES:

1. "skills" must contain skills actually present in the resume.
2. "strengths" must be based on information actually present in the resume.
3. "weaknesses" should identify realistic areas for improvement.
4. "missing_skills" should contain useful skills that are not clearly demonstrated.
5. "recommended_roles" must be realistic for this candidate.
6. "match_score" MUST be a number between 0 and 100.
7. Do not invent education, certificates, jobs, internships or experience.
8. If something is not available, return an empty list [].
9. Return valid JSON only.

RESUME:

{resume_text[:12000]}
"""

    response = ask_ai(
        prompt,
        json_mode=True
    )

    # --------------------------------------------------------
    # Ollama error
    # --------------------------------------------------------

    if response.startswith("__AI_ERROR__"):

        return {
            "error": response.replace(
                "__AI_ERROR__",
                "",
                1
            ).strip()
        }

    # --------------------------------------------------------
    # Empty AI response
    # --------------------------------------------------------

    if not response.strip():

        return {
            "error": (
                "Ollama returned an empty response. "
                "Make sure llama3.2 is installed and running."
            )
        }

    # --------------------------------------------------------
    # Convert AI response to JSON
    # --------------------------------------------------------

    analysis = extract_json(
        response
    )

    # --------------------------------------------------------
    # Invalid JSON
    # --------------------------------------------------------

    if not analysis:

        return {
            "error": (
                "The AI returned an invalid analysis format."
            )
        }

    # --------------------------------------------------------
    # Safely get list values
    # --------------------------------------------------------

    def safe_list(value):

        if isinstance(value, list):

            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if isinstance(value, str):

            if value.strip():
                return [
                    value.strip()
                ]

        return []

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = analysis.get(
        "summary",
        ""
    )

    if not isinstance(
        summary,
        str
    ):

        summary = str(
            summary
        )

    summary = summary.strip()

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    raw_score = analysis.get(
        "match_score",
        0
    )

    try:

        score_text = str(
            raw_score
        )

        score_match = re.search(
            r"\d+(?:\.\d+)?",
            score_text
        )

        if score_match:

            score = float(
                score_match.group()
            )

        else:

            score = 0

    except Exception:

        score = 0

    score = max(
        0,
        min(
            100,
            score
        )
    )

    if score.is_integer():

        score = int(score)

    # --------------------------------------------------------
    # Final clean analysis
    # --------------------------------------------------------

    return {
        "summary": summary
        if summary
        else "No summary generated.",

        "skills": safe_list(
            analysis.get(
                "skills",
                []
            )
        ),

        "strengths": safe_list(
            analysis.get(
                "strengths",
                []
            )
        ),

        "weaknesses": safe_list(
            analysis.get(
                "weaknesses",
                []
            )
        ),

        "missing_skills": safe_list(
            analysis.get(
                "missing_skills",
                []
            )
        ),

        "recommended_roles": safe_list(
            analysis.get(
                "recommended_roles",
                []
            )
        ),

        "match_score": score
    }

# ============================================================
# JOB MATCHING
# ============================================================

def find_matching_jobs(resume_text):

    if not resume_text or not resume_text.strip():
        return []

    try:

        embedding = create_embedding(
            resume_text
        )

        if not embedding:
            return []

        total_jobs = job_collection.count()

        if total_jobs == 0:
            return []

        number_to_query = min(
            10,
            total_jobs
        )

        results = job_collection.query(
            query_embeddings=[embedding],
            n_results=number_to_query,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        matches = []

        ids = results.get(
            "ids",
            [[]]
        )

        documents = results.get(
            "documents",
            [[]]
        )

        metadatas = results.get(
            "metadatas",
            [[]]
        )

        distances = results.get(
            "distances",
            [[]]
        )

        ids = ids[0] if ids else []
        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []

        for i in range(len(ids)):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                and metadatas[i]
                else {}
            )

            description = (
                documents[i]
                if i < len(documents)
                else ""
            )

            distance = (
                distances[i]
                if i < len(distances)
                else 1
            )

            try:

                # ChromaDB cosine distance for normalized
                # embeddings is converted to a percentage.
                similarity = max(
                    0,
                    min(
                        1,
                        1 - float(distance)
                    )
                )

            except Exception:

                similarity = 0

            matches.append({
                "id": ids[i],
                "title": metadata.get(
                    "title",
                    "Unknown Role"
                ),
                "company": metadata.get(
                    "company",
                    "Unknown Company"
                ),
                "description": description,
                "similarity": round(
                    similarity * 100,
                    2
                )
            })

        return matches

    except Exception as e:

        st.error(
            f"Job matching error: {str(e)}"
        )

        return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Resume Analyzer",
            "Job Matching",
            "Career Recommendations",
            "Learning Roadmap",
            "Resume Improver",
            "Interview Preparation",
            "Career Assistant",
            "Database"
        ]
    )

    st.markdown("---")

    st.caption(
        "AI-powered career intelligence platform"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    resume_records = get_records(
        "resume"
    )

    job_records = get_records(
        "job"
    )

    resume_count = len(
        resume_records
    )

    job_count = len(
        job_records
    )

    latest_analysis = st.session_state.get(
        "latest_analysis",
        {}
    )

    analysis_exists = bool(
        latest_analysis
    )

    matches_exist = bool(
        st.session_state.get(
            "job_matches",
            []
        )
    )

    recommendations_exist = bool(
        st.session_state.get(
            "career_recommendations",
            ""
        )
    )

    completed_steps = sum([
        resume_count > 0,
        analysis_exists,
        matches_exist,
        recommendations_exist
    ])

    progress = int(
        (completed_steps / 4) * 100
    )

    skills = (
        latest_analysis.get(
            "skills",
            []
        )
        if isinstance(
            latest_analysis,
            dict
        )
        else []
    )

    missing_skills = (
        latest_analysis.get(
            "missing_skills",
            []
        )
        if isinstance(
            latest_analysis,
            dict
        )
        else []
    )

    recommended_roles = (
        latest_analysis.get(
            "recommended_roles",
            []
        )
        if isinstance(
            latest_analysis,
            dict
        )
        else []
    )

    ai_score = (
        latest_analysis.get(
            "match_score"
        )
        if isinstance(
            latest_analysis,
            dict
        )
        else None
    )

    # ------------------------------------------------------------
    # PREMIUM HERO
    # ------------------------------------------------------------

    st.markdown(f"""
    <div class="dashboard-hero">
        <div class="hero-content">
            <div class="hero-eyebrow">✦ AI-POWERED CAREER INTELLIGENCE</div>
            <div class="hero-title">
                Turn your resume into your<br>
                <span class="hero-highlight">next career opportunity.</span>
            </div>
            <div class="hero-description">
                An intelligent career workspace that analyzes your resume,
                discovers relevant opportunities, identifies skill gaps and
                helps you become interview-ready — all in one place.
            </div>
            <div class="hero-pills">
                <span class="hero-pill">🧠 AI Resume Analysis</span>
                <span class="hero-pill">💼 Semantic Job Matching</span>
                <span class="hero-pill">🎯 Career Intelligence</span>
                <span class="hero-pill">🚀 Personalized Growth</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # LIVE PLATFORM OVERVIEW
    # ------------------------------------------------------------

    st.markdown(
        '<div class="dashboard-section-title">Platform Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-section-subtitle">'
        'Live information from your current project database and AI session.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📄</div>
            <div class="kpi-label">Resumes</div>
            <div class="kpi-value">{resume_count}</div>
            <div class="kpi-caption">Stored candidate profiles</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💼</div>
            <div class="kpi-label">Job Opportunities</div>
            <div class="kpi-value">{job_count}</div>
            <div class="kpi-caption">Jobs in your database</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        skill_count = len(
            skills
        )

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🧩</div>
            <div class="kpi-label">Skills Detected</div>
            <div class="kpi-value">{skill_count}</div>
            <div class="kpi-caption">From latest AI analysis</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:

        if ai_score is not None:

            score_value = (
                f"{ai_score}/100"
            )

            score_caption = (
                "Latest AI resume score"
            )

        else:

            score_value = "—"

            score_caption = (
                "Analyze a resume to generate"
            )

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">⭐</div>
            <div class="kpi-label">AI Resume Score</div>
            <div class="kpi-value">{score_value}</div>
            <div class="kpi-caption">{score_caption}</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # CAREER INTELLIGENCE CENTER
    # ------------------------------------------------------------

    st.markdown(
        '<div class="dashboard-section-title">Career Intelligence Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-section-subtitle">'
        'A quick view of what your AI career workspace knows right now.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        if not resume_count:

            intelligence_title = (
                "Your career workspace is ready"
            )

            intelligence_text = (
                "Start by uploading a resume. The platform will extract your profile "
                "and make it available for AI analysis, job matching and career guidance."
            )

            insight = (
                "💡 Next step: Open Resume Analyzer and upload your latest PDF resume."
            )

        elif not analysis_exists:

            intelligence_title = (
                "Resume connected — analysis is next"
            )

            intelligence_text = (
                "Your resume is stored successfully. Run AI Resume Analysis to extract "
                "skills, strengths, weaknesses, missing skills and recommended roles."
            )

            insight = (
                "💡 Recommended action: Analyze your resume before exploring matches."
            )

        else:

            role_text = (
                recommended_roles[0]
                if recommended_roles
                else "your target career role"
            )

            intelligence_title = (
                "Your profile has AI insights"
            )

            intelligence_text = (
                f"The latest analysis has identified {len(skills)} skill(s) and "
                f"{len(missing_skills)} potential skill gap(s). "
                f"Your strongest direction currently points toward {role_text}."
            )

            insight = (
                "🎯 Next step: Compare your profile with matching jobs and work on the identified gaps."
            )

        st.markdown(f"""
        <div class="intelligence-card">
            <div class="intelligence-label">✦ AI CAREER INSIGHT</div>
            <div class="intelligence-title">{intelligence_title}</div>
            <div class="intelligence-text">{intelligence_text}</div>
            <div class="insight-box">{insight}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:

        if progress == 0:

            status_title = (
                "Ready to begin"
            )

            status_text = (
                "Upload your resume to start building your AI career profile."
            )

        elif progress < 50:

            status_title = (
                "Profile in progress"
            )

            status_text = (
                "Your resume is connected. Continue with AI analysis to unlock more intelligence."
            )

        elif progress < 100:

            status_title = (
                "Career profile developing"
            )

            status_text = (
                "You have started your AI workflow. Continue matching, recommendations and career preparation."
            )

        else:

            status_title = (
                "Career workflow active"
            )

            status_text = (
                "Your main AI workflow steps have been completed in this session."
            )

        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-label">CAREER WORKFLOW</div>
            <div class="profile-title">{status_title}</div>
            <div class="profile-description">{status_text}</div>
            <div class="profile-progress">
                <div class="profile-progress-fill" style="width:{progress}%;"></div>
            </div>
            <div class="profile-progress-text">
                {completed_steps} of 4 core workflow stages completed · {progress}% workflow progress
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SKILL INTELLIGENCE + NEXT ACTIONS
    # ------------------------------------------------------------

    st.markdown(
        '<div class="dashboard-section-title">Profile Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-section-subtitle">'
        'Skills and recommended actions based on the latest available analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    skill_col, action_col = st.columns(
        [1.2, 1]
    )

    with skill_col:

        if skills:

            chips = "".join(
                f'<span class="skill-chip">{str(skill)}</span>'
                for skill in skills[:18]
            )

            if len(skills) > 18:

                skill_extra = (
                    f"<div style='margin-top:9px;color:#94a3b8;font-size:10px;'>"
                    f"+ {len(skills)-18} more skills detected"
                    f"</div>"
                )

            else:

                skill_extra = ""

        else:

            chips = ""

            skill_extra = (
                '<div class="empty-skill">'
                'No AI-detected skills yet. Analyze a resume to populate this section with real resume data.'
                '</div>'
            )

        st.markdown(f"""
        <div class="skill-panel">
            <div style="font-size:15px;font-weight:800;color:#1e293b;">
                🧠 Detected Skill Profile
            </div>
            <div style="margin-top:4px;color:#94a3b8;font-size:10px;">
                Generated from your latest AI resume analysis
            </div>
            <div style="margin-top:11px;">
                {chips}
            </div>
            {skill_extra}
        </div>
        """, unsafe_allow_html=True)

    with action_col:

        if not resume_count:

            actions = [
                (
                    "📄",
                    "Upload your resume",
                    "Add your latest PDF to begin."
                ),
                (
                    "🧠",
                    "Run AI analysis",
                    "Extract skills and profile insights."
                ),
                (
                    "💼",
                    "Explore job matching",
                    "Compare your profile with jobs."
                )
            ]

        elif not analysis_exists:

            actions = [
                (
                    "🧠",
                    "Analyze your resume",
                    "Unlock strengths, gaps and roles."
                ),
                (
                    "💼",
                    "Prepare job matching",
                    "Use your analyzed profile for matching."
                ),
                (
                    "📚",
                    "Plan your growth",
                    "Build a roadmap around your target role."
                )
            ]

        else:

            actions = [
                (
                    "💼",
                    "Find matching jobs",
                    "Use semantic matching to discover opportunities."
                ),
                (
                    "🧩",
                    "Review skill gaps",
                    f"You currently have {len(missing_skills)} identified gap(s)."
                ),
                (
                    "🎤",
                    "Prepare for interviews",
                    "Practice technical, HR and scenario questions."
                )
            ]

        items = ""

        for icon, name, desc in actions:

            items += f"""
            <div class="action-item">
                <div class="action-icon">{icon}</div>
                <div>
                    <div class="action-name">{name}</div>
                    <div class="action-desc">{desc}</div>
                </div>
            </div>
            """

        st.markdown(
            f"""
            <div class="action-panel">
                <div style="font-size:15px;font-weight:800;color:#1e293b;">
                    ⚡ Recommended Next Actions
                </div>
                <div style="margin-top:7px;">
                    {items}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------------------------------------
    # CAREER JOURNEY
    # ------------------------------------------------------------

    st.markdown(
        '<div class="dashboard-section-title">Your AI Career Journey</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-section-subtitle">'
        'A simple path from resume upload to interview preparation.'
        '</div>',
        unsafe_allow_html=True
    )

    journey_states = [
        resume_count > 0,
        analysis_exists,
        matches_exist,
        bool(missing_skills) or recommendations_exist,
        False
    ]

    journey_labels = [
        "Resume",
        "Analyze",
        "Match Jobs",
        "Improve",
        "Interview"
    ]

    journey_html = (
        '<div class="journey-panel">'
        '<div class="journey-wrap">'
    )

    for i, (label, done) in enumerate(
        zip(
            journey_labels,
            journey_states
        ),
        start=1
    ):

        circle_class = (
            "journey-circle done"
            if done
            else "journey-circle"
        )

        symbol = (
            "✓"
            if done
            else str(i)
        )

        journey_html += f"""
        <div class="journey-step">
            <div class="{circle_class}">{symbol}</div>
            <div class="journey-name">{label}</div>
        </div>
        """

        if i < len(journey_labels):

            journey_html += (
                '<div class="journey-line"></div>'
            )

    journey_html += (
        '</div></div>'
    )

    st.markdown(
        journey_html,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------
    # AI TOOLKIT
    # ------------------------------------------------------------

    st.markdown(
        '<div class="dashboard-section-title">AI Career Toolkit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-section-subtitle">'
        'Everything you need to move from profile building to interview preparation.'
        '</div>',
        unsafe_allow_html=True
    )

    tools_data = [
        (
            "🔍",
            "Resume Analyzer",
            "Extract skills, strengths, weaknesses and career signals."
        ),
        (
            "💼",
            "Job Matching",
            "Find roles using semantic similarity between resumes and jobs."
        ),
        (
            "🚀",
            "Career Recommendations",
            "Discover career directions based on your profile."
        ),
        (
            "📚",
            "Learning Roadmap",
            "Create a practical learning path for your target role."
        ),
        (
            "✨",
            "Resume Improver",
            "Improve ATS compatibility and professional wording."
        ),
        (
            "🎤",
            "Interview Preparation",
            "Generate technical, HR and scenario-based practice."
        ),
    ]

    t1, t2, t3 = st.columns(3)

    columns = [
        t1,
        t2,
        t3
    ]

    for index, (
        icon,
        title,
        description
    ) in enumerate(tools_data):

        with columns[index % 3]:

            st.markdown(f"""
            <div class="tool-card">
                <div class="tool-icon">{icon}</div>
                <div class="tool-title">{title}</div>
                <div class="tool-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # DASHBOARD FOOTER
    # ------------------------------------------------------------

    st.markdown("""
    <div class="dashboard-footer">
        <b>Intelligent Resume & Job Matching Platform</b>
        &nbsp;•&nbsp; AI-powered career intelligence
        &nbsp;•&nbsp; Built for smarter career decisions
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RESUME ANALYZER
# ============================================================

elif page == "Resume Analyzer":

    st.title("📄 Resume Analyzer")

    st.write(
        "Upload your resume and let AI analyze your profile."
    )

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        key="resume_uploader"
    )

    if uploaded_file:

        # --------------------------------------------------------
        # SAVE PDF
        # --------------------------------------------------------

        file_path = os.path.join(
            RESUME_DIR,
            uploaded_file.name
        )

        try:

            file_bytes = uploaded_file.getvalue()

            with open(
                file_path,
                "wb"
            ) as f:

                f.write(file_bytes)

        except Exception as e:

            st.error(
                f"Could not save the resume: {str(e)}"
            )

            st.stop()

        # --------------------------------------------------------
        # EXTRACT TEXT
        # --------------------------------------------------------

        try:

            resume_text = extract_pdf_text(
                file_path
            )

        except Exception as e:

            st.error(
                f"Could not read the PDF: {str(e)}"
            )

            resume_text = ""

        if not resume_text:

            st.error(
                "Could not extract text from this PDF. "
                "Please upload a text-based PDF rather than a scanned image PDF."
            )

        else:

            st.success(
                "Resume uploaded successfully."
            )

            # Show extracted text preview
            with st.expander(
                "View extracted resume text"
            ):

                st.text(
                    resume_text[:5000]
                )

            # ----------------------------------------------------
            # ANALYZE BUTTON
            # ----------------------------------------------------

            if st.button(
                "🤖 Analyze Resume",
                type="primary",
                key="analyze_resume_button"
            ):

                with st.spinner(
                    "AI is analyzing your resume..."
                ):

                    try:

                        # Step 1:
                        # Analyze with Ollama
                        analysis = analyze_resume(
                            resume_text
                        )

                        # Step 2:
                        # Validate result
                        if not analysis:

                            st.error(
                                "The AI did not return a valid analysis. "
                                "Please check that Ollama and llama3.2 are running."
                            )

                        elif (
                            analysis.get("summary", "")
                            .startswith("AI Error:")
                        ):

                            st.error(
                                analysis["summary"]
                            )

                        else:

                            # Step 3:
                            # Store resume in ChromaDB
                            add_resume_to_database(
                                uploaded_file.name,
                                uploaded_file.name,
                                resume_text
                            )

                            # Step 4:
                            # Store analysis in session
                            st.session_state[
                                "latest_analysis"
                            ] = analysis

                            # Clear old matching results
                            st.session_state[
                                "job_matches"
                            ] = []

                            st.success(
                                "Resume analysis completed successfully!"
                            )

                    except Exception as e:

                        st.error(
                            f"Resume analysis failed: {str(e)}"
                        )
    # ========================================================
    # DISPLAY ANALYSIS
    # ========================================================

    analysis = st.session_state.get(
        "latest_analysis",
        {}
    )

    if analysis:

        st.subheader(
            "AI Resume Analysis"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "### 📝 Summary"
        )

        summary_text = analysis.get(
            "summary",
            "Not available"
        )

        st.markdown(
            f"""
            <div style="
                color:#172033;
                background:#ffffff;
                border:1px solid #e2e8f0;
                border-radius:12px;
                padding:15px 18px;
                font-size:14px;
                line-height:1.7;
                margin-bottom:20px;
            ">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # SKILLS + STRENGTHS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🧠 Skills"
            )

            skills = analysis.get(
                "skills",
                []
            )

            if skills:

                skills_html = ""

                for skill in skills:

                    skills_html += f"""
                    <span style="
                        display:inline-block;
                        background:#eef2ff;
                        color:#1e1b4b;
                        border:1px solid #c7d2fe;
                        border-radius:8px;
                        padding:7px 11px;
                        margin:4px 5px 4px 0;
                        font-size:13px;
                        font-weight:600;
                    ">
                        {skill}
                    </span>
                    """

                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        border:1px solid #e2e8f0;
                        border-radius:12px;
                        padding:14px;
                        min-height:70px;
                    ">
                        {skills_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div style="
                        color:#475569;
                        background:#f8fafc;
                        border:1px solid #e2e8f0;
                        border-radius:12px;
                        padding:14px;
                    ">
                        No skills detected.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:

            st.markdown(
                "### 💪 Strengths"
            )

            strengths = analysis.get(
                "strengths",
                []
            )

            if strengths:

                strengths_html = ""

                for strength in strengths:

                    strengths_html += f"""
                    <div style="
                        color:#172033;
                        background:#ffffff;
                        border:1px solid #e2e8f0;
                        border-radius:9px;
                        padding:9px 12px;
                        margin-bottom:7px;
                        font-size:13px;
                        line-height:1.5;
                    ">
                        ✓ {strength}
                    </div>
                    """

                st.markdown(
                    strengths_html,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div style="
                        color:#475569;
                        background:#f8fafc;
                        border:1px solid #e2e8f0;
                        border-radius:12px;
                        padding:14px;
                    ">
                        No strengths detected.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # WEAKNESSES
        # ----------------------------------------------------

        st.markdown(
            "### ⚠️ Weaknesses"
        )

        weaknesses = analysis.get(
            "weaknesses",
            []
        )

        if weaknesses:

            weaknesses_html = ""

            for weakness in weaknesses:

                weaknesses_html += f"""
                <div style="
                    color:#172033;
                    background:#ffffff;
                    border:1px solid #e2e8f0;
                    border-radius:9px;
                    padding:10px 13px;
                    margin-bottom:7px;
                    font-size:13px;
                    line-height:1.5;
                ">
                    • {weakness}
                </div>
                """

            st.markdown(
                weaknesses_html,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style="
                    color:#475569;
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:12px;
                    padding:14px;
                ">
                    No weaknesses identified.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # MISSING SKILLS
        # ----------------------------------------------------

        st.markdown(
            "### 📚 Missing Skills"
        )

        missing_skills = analysis.get(
            "missing_skills",
            []
        )

        if missing_skills:

            missing_html = ""

            for skill in missing_skills:

                missing_html += f"""
                <span style="
                    display:inline-block;
                    color:#172033;
                    background:#fff7ed;
                    border:1px solid #fed7aa;
                    border-radius:8px;
                    padding:7px 11px;
                    margin:4px 5px 4px 0;
                    font-size:13px;
                    font-weight:600;
                ">
                    {skill}
                </span>
                """

            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #e2e8f0;
                    border-radius:12px;
                    padding:14px;
                ">
                    {missing_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style="
                    color:#475569;
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:12px;
                    padding:14px;
                ">
                    No major missing skills identified.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # RECOMMENDED ROLES
        # ----------------------------------------------------

        st.markdown(
            "### 🎯 Recommended Roles"
        )

        recommended_roles = analysis.get(
            "recommended_roles",
            []
        )

        if recommended_roles:

            roles_html = ""

            for role in recommended_roles:

                roles_html += f"""
                <div style="
                    color:#172033;
                    background:#ffffff;
                    border:1px solid #e2e8f0;
                    border-radius:10px;
                    padding:10px 13px;
                    margin-bottom:7px;
                    font-size:13px;
                    font-weight:600;
                ">
                    🎯 {role}
                </div>
                """

            st.markdown(
                roles_html,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style="
                    color:#475569;
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:12px;
                    padding:14px;
                ">
                    No recommended roles generated.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = analysis.get(
            "match_score",
            0
        )

        try:

            score = float(score)

        except Exception:

            score = 0

        score = max(
            0,
            min(
                100,
                score
            )
        )

        if score.is_integer():

            score_display = int(score)

        else:

            score_display = round(
                score,
                1
            )

        st.markdown("### ⭐ Resume Score")

        st.metric(
            label="AI Resume Score",
            value=f"{score_display}/100"
        )

# ============================================================
# JOB MATCHING
# ============================================================

elif page == "Job Matching":

    st.title(
        "💼 Intelligent Job Matching"
    )

    resumes = get_records(
        "resume"
    )

    if not resumes:

        st.info(
            "Please upload and analyze a resume first."
        )

    else:

        selected_resume = st.selectbox(
            "Select Resume",
            resumes,
            format_func=lambda x:
            x["metadata"].get(
                "name",
                x["id"]
            )
        )

        if st.button(
            "🔎 Find Matching Jobs",
            type="primary"
        ):

            with st.spinner(
                "Finding suitable jobs..."
            ):

                matches = find_matching_jobs(
                    selected_resume["document"]
                )

                st.session_state[
                    "job_matches"
                ] = matches

        matches = st.session_state.get(
            "job_matches",
            []
        )

        if matches:

            for match in matches:

                score = match.get(
                    "similarity",
                    0
                )

                with st.container(
                    border=True
                ):

                    st.subheader(
                        match.get(
                            "title",
                            "Unknown Role"
                        )
                    )

                    st.write(
                        f"🏢 {match.get('company', 'Unknown Company')}"
                    )

                    st.progress(
                        min(
                            max(
                                float(score) / 100,
                                0
                            ),
                            1.0
                        )
                    )

                    st.write(
                        f"Match Score: **{score}%**"
                    )

                    st.write(
                        match.get(
                            "description",
                            ""
                        )[:700]
                    )

        elif st.session_state.get(
            "job_matches"
        ) == []:

            if job_collection.count() == 0:

                st.info(
                    "No jobs are currently stored in the database. "
                    "Add jobs to the Job Database before matching."
                )


# ============================================================
# CAREER RECOMMENDATIONS
# ============================================================

elif page == "Career Recommendations":

    st.title(
        "🚀 Career Recommendations"
    )

    resumes = get_records(
        "resume"
    )

    if not resumes:

        st.info(
            "Upload a resume to receive recommendations."
        )

    else:

        selected_resume = st.selectbox(
            "Select Resume",
            resumes,
            format_func=lambda x:
            x["metadata"].get(
                "name",
                x["id"]
            )
        )

        if st.button(
            "Generate Career Recommendations",
            type="primary"
        ):

            prompt = f"""
Analyze this resume and recommend the most
suitable career roles.

Return:

1. Top career roles
2. Why each role is suitable
3. Required skills
4. Skills to improve

Do not invent qualifications.

Resume:

{selected_resume["document"][:9000]}
"""

            with st.spinner(
                "Generating recommendations..."
            ):

                result = ask_ai(
                    prompt
                )

                st.session_state[
                    "career_recommendations"
                ] = result

        recommendation = st.session_state.get(
            "career_recommendations",
            ""
        )

        if recommendation:

            st.markdown(
                recommendation
            )


# ============================================================
# LEARNING ROADMAP
# ============================================================

elif page == "Learning Roadmap":

    st.title(
        "📚 Personalized Learning Roadmap"
    )

    role = st.text_input(
        "Enter your target career role",
        placeholder="Example: Data Scientist"
    )

    current_skills = st.text_area(
        "Enter your current skills",
        placeholder="Python, SQL, Statistics..."
    )

    if st.button(
        "Generate Roadmap",
        type="primary"
    ):

        if role.strip() and current_skills.strip():

            prompt = f"""
Create a practical learning roadmap for becoming:

{role}

Current skills:

{current_skills}

Include:

1. Skills to learn
2. Learning order
3. Projects
4. Tools
5. Practice activities
6. Interview preparation
7. Suggested timeline

Keep the roadmap suitable for a college student.
"""

            with st.spinner(
                "Creating your roadmap..."
            ):

                roadmap = ask_ai(
                    prompt
                )

                st.markdown(
                    roadmap
                )

        else:

            st.warning(
                "Please enter both target role and current skills."
            )


# ============================================================
# RESUME IMPROVER
# ============================================================

elif page == "Resume Improver":

    st.title(
        "✨ AI Resume Improver"
    )

    resumes = get_records(
        "resume"
    )

    if not resumes:

        st.info(
            "Upload a resume first."
        )

    else:

        selected_resume = st.selectbox(
            "Select Resume",
            resumes,
            format_func=lambda x:
            x["metadata"].get(
                "name",
                x["id"]
            )
        )

        if st.button(
            "✨ Improve Resume",
            type="primary"
        ):

            prompt = f"""
Improve the following resume.

Focus on:

- ATS compatibility
- Strong action verbs
- Clear achievements
- Professional wording
- Relevant keywords
- Better structure

Do not invent qualifications.

Resume:

{selected_resume["document"][:10000]}
"""

            with st.spinner(
                "Improving resume..."
            ):

                improved = ask_ai(
                    prompt
                )

                st.markdown(
                    improved
                )


# ============================================================
# INTERVIEW PREPARATION
# ============================================================

elif page == "Interview Preparation":

    st.title(
        "🎤 AI Interview Preparation"
    )

    role = st.text_input(
        "Target Job Role",
        placeholder="Example: Data Analyst"
    )

    level = st.selectbox(
        "Experience Level",
        [
            "Fresher",
            "Entry Level",
            "Intermediate"
        ]
    )

    if st.button(
        "Generate Interview Questions",
        type="primary"
    ):

        if role.strip():

            prompt = f"""
Generate interview preparation material for:

Role: {role}
Level: {level}

Include:

1. 5 technical questions
2. 5 HR questions
3. 3 scenario-based questions
4. Important topics to revise
5. Tips for answering
"""

            with st.spinner(
                "Preparing interview questions..."
            ):

                result = ask_ai(
                    prompt
                )

                st.markdown(
                    result
                )

        else:

            st.warning(
                "Enter a target job role."
            )


# ============================================================
# CAREER ASSISTANT
# ============================================================

elif page == "Career Assistant":

    st.title(
        "🤖 AI Career Assistant"
    )

    st.write(
        "Ask anything related to resumes, careers, jobs, skills or interviews."
    )

    for message in st.session_state[
        "career_messages"
    ]:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    user_question = st.chat_input(
        "Ask your career question..."
    )

    if user_question:

        st.session_state[
            "career_messages"
        ].append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_question
            )

        prompt = f"""
You are an AI career assistant.

Give practical, clear and student-friendly
career advice.

User question:

{user_question}
"""

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_ai(
                    prompt
                )

                st.markdown(
                    answer
                )

        st.session_state[
            "career_messages"
        ].append({
            "role": "assistant",
            "content": answer
        })


# ============================================================
# DATABASE
# ============================================================

elif page == "Database":

    st.title(
        "🗄️ Project Database"
    )

    tab1, tab2 = st.tabs(
        [
            "📄 Resumes",
            "💼 Jobs"
        ]
    )

    # --------------------------------------------------------
    # RESUMES
    # --------------------------------------------------------

    with tab1:

        resumes = get_records(
            "resume"
        )

        st.subheader(
            f"Stored Resumes ({len(resumes)})"
        )

        if not resumes:

            st.info(
                "No resumes stored."
            )

        else:

            for resume in resumes:

                metadata = resume[
                    "metadata"
                ]

                file_name = metadata.get(
                    "file_name",
                    ""
                )

                display_name = metadata.get(
                    "name",
                    file_name
                )

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    st.write(
                        f"📄 **{display_name}**"
                    )

                with col2:

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_resume_{resume['id']}"
                    ):

                        delete_resume(
                            resume["id"],
                            file_name
                        )

                        # Clear current analysis if
                        # the deleted resume was being shown.
                        st.session_state[
                            "latest_analysis"
                        ] = {}

                        st.session_state[
                            "job_matches"
                        ] = []

                        st.success(
                            "Resume deleted successfully."
                        )

                        st.rerun()

    # --------------------------------------------------------
    # JOBS
    # --------------------------------------------------------

    with tab2:

        # ====================================================
        # ADD NEW JOB
        # ====================================================

        st.subheader("➕ Add New Job")

        job_title = st.text_input(
            "Job Title",
            placeholder="Example: Data Scientist Intern",
            key="job_title_input"
        )

        job_company = st.text_input(
            "Company",
            placeholder="Example: ABC Technologies",
            key="job_company_input"
        )

        job_description = st.text_area(
            "Job Description",
            placeholder=(
                "Enter the complete job description including "
                "required skills, qualifications and responsibilities."
            ),
            height=180,
            key="job_description_input"
        )

        if st.button(
            "💼 Add Job to Database",
            type="primary",
            key="add_job_button"
        ):

            if not job_title.strip():

                st.warning(
                    "Please enter a Job Title."
                )

            elif not job_company.strip():

                st.warning(
                    "Please enter the Company name."
                )

            elif not job_description.strip():

                st.warning(
                    "Please enter the Job Description."
                )

            else:

                try:

                    add_job_to_database(
                        job_title.strip(),
                        job_company.strip(),
                        job_description.strip()
                    )

                    st.success(
                        f"Job '{job_title}' added successfully to the database!"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Could not add job: {str(e)}"
                    )

        st.markdown("---")

        # ====================================================
        # STORED JOBS
        # ====================================================

        jobs = get_records(
            "job"
        )

        st.subheader(
            f"Stored Jobs ({len(jobs)})"
        )

        if not jobs:

            st.info(
                "No jobs stored."
            )

        else:

            for job in jobs:

                metadata = job[
                    "metadata"
                ]

                title = metadata.get(
                    "title",
                    "Unknown Role"
                )

                company = metadata.get(
                    "company",
                    "Unknown Company"
                )

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    st.write(
                        f"💼 **{title}** — {company}"
                    )

                with col2:

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_job_{job['id']}"
                    ):

                        delete_job(
                            job["id"]
                        )

                        st.session_state[
                            "job_matches"
                        ] = []

                        st.success(
                            "Job deleted successfully."
                        )

                        st.rerun()