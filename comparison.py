"""
Medical RAG Evaluation — Local Variant
═══════════════════════════════════════════════════════════════════
Hardware   : NVIDIA RTX 5070 Ti  (Blackwell / GB203, 16 GB GDDR7)
CUDA       : 12.8  →  install with cu128 wheels
Python     : 3.10+

Install
───────
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install sentence-transformers transformers faiss-cpu
pip install pandas numpy matplotlib seaborn tqdm psutil

Data files  (all in DATA_DIR — default: ./data/)
────────────────────────────────────────────────
  mtsamples.csv          Kaggle medical transcriptions (Tara Boyle)
                         cols: description, medical_specialty, sample_name,
                               transcription, keywords
  X.csv                  4-class processed corpus (from Data Modelling notebook)
                         cols: label (1-4), description, text
  classes.txt            Label names for X.csv
                         1=Surgery  2=Medical Records
                         3=Internal Medicine  4=Other
  clinical-stopwords.txt Clinical + general stopwords (Dr. Kavita Ganesan)
  vocab.txt              SNOMED-derived vocabulary (69,944 tokens, one per line)
                         Used for: tokenizer probe set, SNOMED coverage scoring,
                         and SNOMED-boosted key-term extraction
"""

# ═══════════════════════════════════════════════════════════════
#  SECTION 0 — USER CONFIG
# ═══════════════════════════════════════════════════════════════

import warnings
import random
import time
import gc
import re
import torch
import faiss
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import psutil
import pandas as pd
import numpy as np
import os

DATA_DIR = "./data"      # folder containing all five input files
CACHE_DIR = "./hf_cache"  # HuggingFace model / tokenizer cache
OUT_DIR = "./results"   # all output files land here

MTS_FILE = os.path.join(DATA_DIR, "mtsamples.csv")
X_FILE = os.path.join(DATA_DIR, "X.csv")
CLS_FILE = os.path.join(DATA_DIR, "classes.txt")
SW_FILE = os.path.join(DATA_DIR, "clinical-stopwords.txt")
VOC_FILE = os.path.join(DATA_DIR, "vocab.txt")

MAX_DOCS = 10_000   # cap on combined unique documents
USE_FP16 = True     # half-precision — safe on all listed models

# RTX 5070 Ti — 16 GB GDDR7, Blackwell arch (sm_120)
BATCH_SIZES = {
    "small":  512,
    "base":   256,
    "large":  128,
}

# Number of SNOMED terms to sample for the per-term tokenizer heatmap
SNOMED_PROBE_N = 60   # keep heatmap legible; full coverage uses all 69 k

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — IMPORTS & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════


matplotlib.use("Agg")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

for d in (CACHE_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

# ── colours ────────────────────────────────────────────────────
C_GEN = "#4C8EDA"   # blue  — general-purpose models
C_MED = "#E8593C"   # coral — medical domain-adapted models
C_MAP = {"general": C_GEN, "medical": C_MED}
BG = "#F8F7F4"
BG_AX = "#FFFFFF"
GRID = "#E8E6E0"
TEXT = "#2C2C2A"
TEXT_S = "#5F5E5A"
SPEC_P = sns.color_palette("tab20", 20)

sns.set_theme(
    style="whitegrid",
    rc={
        "figure.facecolor":  BG,
        "axes.facecolor":    BG_AX,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT_S,
        "ytick.color":       TEXT_S,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    },
)

METRIC_COLS = [
    "Avg Prec@5", "Avg Recall@5", "Avg MRR@5", "Avg nDCG@5", "Overall Quality"
]

# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — RESOURCE LOADERS
#  (stopwords, X.csv class labels, SNOMED vocab)
# ═══════════════════════════════════════════════════════════════

# ── module-level singletons (populated in main()) ──────────────
STOPWORDS:       set[str] = set()
X_LABEL_MAP:     dict[int, str] = {}
# full lowercase set  — for coverage scoring
SNOMED_VOCAB:    set[str] = set()
# curated sample      — for tokenizer heatmap
SNOMED_PROBE:    list[str] = []


def load_stopwords(path: str) -> set[str]:
    """
    Parse clinical-stopwords.txt (Dr. Kavita Ganesan).
    Lines beginning with '#' are comments; blank lines skipped.
    """
    if not os.path.exists(path):
        print(f"  [warn] stopwords not found: {path} — using minimal fallback")
        return {
            "the", "and", "or", "of", "in", "to", "a", "an", "is", "was", "are", "were",
            "for", "on", "at", "by", "with", "as", "be", "been", "have", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might",
        }
    sw: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            tok = line.strip()
            if tok and not tok.startswith("#"):
                sw.add(tok.lower())
    print(f"  Loaded {len(sw):,} clinical stopwords")
    return sw


def load_classes(path: str) -> dict[int, str]:
    """
    Parse classes.txt — one class label per line, 1-indexed.
    Expected content:
        Surgery
        Medical Records
        Internal Medicine
        Other
    Returns  {1: 'Surgery', 2: 'Medical Records', 3: 'Internal Medicine', 4: 'Other'}
    """
    if not os.path.exists(path):
        # hard-coded fallback — verified against classes.txt
        print(f"  [warn] classes.txt not found — using hard-coded label map")
        return {1: "Surgery", 2: "Medical Records",
                3: "Internal Medicine", 4: "Other"}
    labels: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            name = line.strip()
            if name:
                labels[idx] = name
    print(f"  Loaded {len(labels)} class labels: {labels}")
    return labels


def load_snomed_vocab(path: str, stopwords: set[str]) -> tuple[set[str], list[str]]:
    """
    Load vocab.txt — SNOMED-derived vocabulary (69,944 tokens, one per line).
    Generated from the Systematized Nomenclature of Medicine International data.

    Returns
    ───────
    vocab_set   – full lowercase set of all tokens (used for coverage scoring
                  and SNOMED-boosted key-term extraction)
    probe_list  – SNOMED_PROBE_N curated alpha terms for tokenizer heatmap;
                  filters: alpha-only, 5-20 chars, Title-case, not in stopwords
    """
    if not os.path.exists(path):
        print(f"  [warn] vocab.txt not found: {path}")
        return set(), []

    with open(path, encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    vocab_set = {t.lower() for t in lines}

    # probe: medical single-word terms suitable for subword analysis
    candidates = [
        t for t in lines
        if t.isalpha()
        and 5 <= len(t) <= 20
        and t[0].isupper()          # Title-case anatomical / clinical terms
        and not t.isupper()         # exclude pure acronyms (HNSHA, NOS…)
        and t.lower() not in stopwords
    ]
    random.seed(42)
    probe = sorted(random.sample(
        candidates, min(SNOMED_PROBE_N, len(candidates))))

    print(f"  Loaded SNOMED vocab: {len(vocab_set):,} tokens  |  "
          f"probe set: {len(probe)} terms")
    return vocab_set, probe


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — DATA LOADING & UNIFIED DATAFRAME
# ═══════════════════════════════════════════════════════════════

def load_mtsamples(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "original_idx"}, errors="ignore")
    df = df[df["transcription"].notna()].copy()
    df = df[df["transcription"].str.strip().str.len() > 20].copy()
    df["text"] = df["transcription"].str.strip()
    df["source"] = "mtsamples"
    df["specialty"] = df["medical_specialty"].str.strip()
    df["description"] = df.get("description",
                               pd.Series("", index=df.index)).fillna("").str.strip()
    df["keywords"] = df.get("keywords",
                            pd.Series("", index=df.index)).fillna("").str.strip()
    df["class_label"] = "—"   # mtsamples has fine-grained specialty, not X classes
    return df[["text", "source", "specialty", "description",
               "keywords", "class_label"]].reset_index(drop=True)


def load_x(path: str) -> pd.DataFrame:
    """
    Uses X_LABEL_MAP (from classes.txt):
        1 → Surgery
        2 → Medical Records
        3 → Internal Medicine
        4 → Other
    """
    df = pd.read_csv(path)
    df = df[df["text"].notna()].copy()
    df = df[df["text"].str.strip().str.len() > 20].copy()
    df["text"] = df["text"].str.strip()
    df["source"] = "x_corpus"
    df["specialty"] = df["label"].map(X_LABEL_MAP).fillna("Unknown")
    df["description"] = df.get("description",
                               pd.Series("", index=df.index)).fillna("").str.strip()
    df["keywords"] = ""
    df["class_label"] = df["label"].map(X_LABEL_MAP).fillna("Unknown")
    return df[["text", "source", "specialty", "description",
               "keywords", "class_label"]].reset_index(drop=True)


def build_corpus(max_docs: int = MAX_DOCS) -> pd.DataFrame:
    frames = []
    if os.path.exists(MTS_FILE):
        mts = load_mtsamples(MTS_FILE)
        frames.append(mts)
        print(f"  mtsamples : {len(mts):,} docs")
    else:
        print(f"  [warn] {MTS_FILE} not found")

    if os.path.exists(X_FILE):
        xdf = load_x(X_FILE)
        frames.append(xdf)
        print(f"  X.csv     : {len(xdf):,} docs  "
              f"({xdf['specialty'].value_counts().to_dict()})")
    else:
        print(f"  [warn] {X_FILE} not found")

    if not frames:
        raise FileNotFoundError(
            f"No data files found in {DATA_DIR}/. "
            "Place mtsamples.csv and X.csv there and re-run."
        )

    corpus = pd.concat(frames, ignore_index=True)

    # Deduplicate on first-150-char fingerprint; mtsamples rows preferred
    corpus["fingerprint"] = corpus["text"].str.strip().str[:150]
    n_before = len(corpus)
    corpus = corpus.drop_duplicates(
        "fingerprint", keep="first").reset_index(drop=True)
    print(f"  Dedup: {n_before:,} → {len(corpus):,} rows "
          f"({n_before - len(corpus):,} dupes removed)")

    corpus = corpus.head(max_docs).reset_index(drop=True)
    corpus["word_count"] = corpus["text"].str.split().str.len()
    corpus["char_count"] = corpus["text"].str.len()

    print(f"\n  Final corpus : {len(corpus):,} docs")
    print(f"    mtsamples  : {(corpus['source']=='mtsamples').sum():,}")
    print(f"    x_corpus   : {(corpus['source']=='x_corpus').sum():,}")
    print(f"    specialties: {corpus['specialty'].nunique()}")
    return corpus


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — CHUNKING
# ═══════════════════════════════════════════════════════════════

def sentence_chunk(text: str, max_words: int = 180) -> list[str]:
    """
    Sentence-boundary aware.
    Also splits on MT-Samples section-header pattern: "HEADING:,  text…"
    """
    segs = re.split(r'(?<=[.!?])\s+|(?<=,)\s{2,}', text.strip())
    chunks, cur = [], ""
    for s in segs:
        s = s.strip()
        if not s:
            continue
        if len(cur.split()) + len(s.split()) > max_words:
            if cur:
                chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if len(c.split()) >= 5]


def fixed_chunk(text: str, window: int = 150, stride: int = 75) -> list[str]:
    """50%-overlap sliding window (word-level)."""
    words = text.split()
    chunks = []
    for start in range(0, len(words), stride):
        c = " ".join(words[start: start + window])
        if len(c.split()) >= 5:
            chunks.append(c)
        if start + window >= len(words):
            break
    return chunks


def build_chunks(corpus: pd.DataFrame):
    s_chunks, f_chunks = [], []
    s_meta,   f_meta = [], []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus),
                       desc="  chunking", ncols=80):
        for c in sentence_chunk(row["text"]):
            s_chunks.append(c)
            s_meta.append({"specialty":   row["specialty"],
                           "class_label": row["class_label"],
                           "source":      row["source"],
                           "word_count":  len(c.split())})
        for c in fixed_chunk(row["text"]):
            f_chunks.append(c)
            f_meta.append({"specialty":   row["specialty"],
                           "class_label": row["class_label"],
                           "source":      row["source"],
                           "word_count":  len(c.split())})

    s_meta = pd.DataFrame(s_meta)
    f_meta = pd.DataFrame(f_meta)
    print(f"\n  Sentence chunks : {len(s_chunks):,}  "
          f"(mean {s_meta['word_count'].mean():.0f} words)")
    print(f"  Fixed chunks    : {len(f_chunks):,}  "
          f"(mean {f_meta['word_count'].mean():.0f} words)")
    return s_chunks, f_chunks, s_meta, f_meta


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — QUERIES
# ═══════════════════════════════════════════════════════════════

QUERIES = [
    "What treatments were recommended for a patient with allergic rhinitis who found Allegra ineffective and has no prescription coverage?",
    "Describe the physical exam findings in the nose and throat for a 23-year-old female with worsening allergies after moving from Seattle.",
    "What nasal spray samples and alternative oral medication were suggested for allergic rhinitis in a patient switching from Allegra?",
    "For a patient with chronic sinusitis, facial pain, postnasal drip and turbinate hypertrophy, what medications and tests were ordered?",
    "What history and recommendations were given for a 55-year-old female with chronic glossitis, xerostomia, probable food and inhalant allergies?",
    "In a case of Kawasaki disease in a 14-month-old, what discharge medications and follow-up instructions were provided after IVIG?",
    "What symptoms and plan were documented for a 42-year-old woman with worsening asthma aggravated by corn hauling and irregular Allegra use?",
    "What surgical steps were described in a laparoscopic antecolic antegastric Roux-en-Y gastric bypass with EEA anastomosis for morbid obesity?",
    "For a 42-year-old male weighing 344 pounds with BMI 51, sleep apnea, diabetes and joint pain, what preoperative workup and risks were discussed?",
    "Describe the eating history and weight loss attempts of a single male patient in a gastric bypass consult who weighs 312 pounds.",
    "What port placement and bowel division technique was used in a 30-year-old female undergoing laparoscopic gastric bypass?",
    "In a bariatric consult, what family history, social habits, and review of systems negatives were noted for a patient pursuing weight loss surgery?",
    "What complications like anastomotic leak, bleeding, DVT, and bowel obstruction were explained to a patient before Roux-en-Y gastric bypass?",
    "What were the key 2-D M-mode and Doppler findings in an echocardiogram showing left atrial enlargement of 4.7 cm and PA systolic pressure of 36 mmHg?",
    "Describe the left ventricular findings in a hyperdynamic echocardiogram with estimated EF 70-75%, near-cavity obliteration, and elevated LA pressures.",
    "In an echo report, what valve abnormalities and pressures were noted with mild aortic stenosis, calcified valve, and moderate biatrial enlargement?",
    "What normal findings were reported in an echocardiogram with no pericardial effusion, normal LV systolic function, and trace mitral/tricuspid regurgitation?",
    "For a patient with mild tricuspid regurgitation and right heart pressures 30-35 mmHg, what was the summary impression of the 2-D study?",
    "What procedure details and anesthesia were used in a liposuction case combined with right breast reconstruction revision after latissimus dorsi flap?",
    "In a suction-assisted lipectomy for abdominal and thigh lipodystrophy, what volume of aspirate was removed and what cannulas were used?",
    "Compare the ejection fraction and chamber sizes reported across different 2-D echocardiogram samples in the dataset.",
    "What common comorbidities (asthma, diabetes, sleep apnea) appear in both allergic rhinitis and morbid obesity gastric bypass patients?",
    "Describe any mention of allergy testing (RAST) or EpiPen prescription in patients with suspected drug or environmental reactions.",
    "What physical exam findings (wheezing, edema, vital signs) were documented in bariatric consults and asthma-related visits?",
    "In cases involving anastomosis (gastric bypass or other), what techniques or tools (EEA stapler, Surgidac sutures) were mentioned?",
]
QUERY_LABELS = [f"Q{i+1:02d}" for i in range(len(QUERIES))]

# Map each query to its X.csv class label (from classes.txt)
QUERY_CLASS = {
    "Surgery":          ["Q08", "Q09", "Q10", "Q11", "Q12", "Q13", "Q19", "Q20", "Q25"],
    "Internal Medicine": ["Q14", "Q15", "Q16", "Q17", "Q18", "Q21"],
    "Medical Records":  ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07"],
    "Other":            ["Q22", "Q23", "Q24"],
}


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════

MODELS: dict[str, tuple[str, str, str]] = {
    # name           (hf_id,                                      type,      tier)
    "MiniLM-L6":    ("sentence-transformers/all-MiniLM-L6-v2",   "general", "small"),
    "bge-small":    ("BAAI/bge-small-en-v1.5",                   "general", "small"),
    "bge-base":     ("BAAI/bge-base-en-v1.5",                    "general", "base"),
    "Snowflake-M":  ("Snowflake/snowflake-arctic-embed-m",        "general", "base"),
    "e5-large":     ("intfloat/e5-large-v2",                     "general", "large"),
    "GTE-Large":    ("thenlper/gte-large",                        "general", "large"),
    "PubMedBERT":   ("NeuML/pubmedbert-base-embeddings",          "medical", "base"),
    "BioBERT-MNLI": ("pritamdeka/S-BioBERT-mnli-snli",            "medical", "base"),
    "MedEmbed-L":   ("abhinand/medembed-large-v0.1",             "medical", "large"),
}
MODEL_ORDER = list(MODELS.keys())


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — TOKENIZER ANALYSIS
#  Three-tier evaluation:
#    A. Probe metrics   — fertility, UNK rate on SNOMED_PROBE terms
#    B. SNOMED coverage — % of full 69 k vocab tokenised as 1 subword
#    C. Per-term table  — raw subword breakdown for heatmap
# ═══════════════════════════════════════════════════════════════

PLAIN_TERMS = [
    "the", "patient", "hospital", "blood", "heart", "pain", "doctor",
    "medication", "surgery", "treatment", "diagnosis", "fever",
    "infection", "test", "result", "care", "report", "status", "normal", "exam",
]


def snomed_coverage(tok: "AutoTokenizer", sample_size: int = 5_000) -> float:
    """
    Sample `sample_size` tokens from SNOMED_VOCAB (alpha, len 4-18).
    Return the fraction that tokenise to exactly 1 subword.
    Higher = better medical vocabulary coverage.
    """
    if not SNOMED_VOCAB:
        return float("nan")
    candidates = [t for t in SNOMED_VOCAB if t.isalpha() and 4 <= len(t) <= 18]
    random.seed(0)
    sample = random.sample(candidates, min(sample_size, len(candidates)))
    single = sum(
        1 for t in sample
        if len(tok.encode(t, add_special_tokens=False)) == 1
    )
    return round(single / len(sample), 4)


def analyze_tokenizer(name: str, path: str, mtype: str
                      ) -> tuple[dict, pd.DataFrame]:
    print(f"    {name:<16s}", end="", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(path, cache_dir=CACHE_DIR)
        vocab_sz = tok.vocab_size
        unk_id = tok.unk_token_id

        # ── fertility (SNOMED probe + plain mix) ──────────────────
        sample = " ".join(SNOMED_PROBE * 2 + PLAIN_TERMS * 5)
        words = sample.split()
        fertility = len(tok.tokenize(sample)) / max(len(words), 1)

        # ── UNK rate on SNOMED probe ───────────────────────────────
        n_unk = n_tot = 0
        for term in SNOMED_PROBE:
            ids = tok.encode(term, add_special_tokens=False)
            n_unk += sum(1 for i in ids if i == unk_id)
            n_tot += len(ids)
        unk_rate = n_unk / max(n_tot, 1)

        # ── avg tokens per probe term vs plain ─────────────────────
        probe_c = [len(tok.encode(t, add_special_tokens=False))
                   for t in SNOMED_PROBE]
        plain_c = [len(tok.encode(t, add_special_tokens=False))
                   for t in PLAIN_TERMS]
        avg_probe = float(np.mean(probe_c))
        avg_plain = float(np.mean(plain_c))

        # ── SNOMED full-vocab coverage ─────────────────────────────
        cov = snomed_coverage(tok)

        # ── per-term breakdown for heatmap ─────────────────────────
        term_rows = []
        for term, n in zip(SNOMED_PROBE, probe_c):
            ids = tok.encode(term, add_special_tokens=False)
            subs = tok.convert_ids_to_tokens(ids)
            term_rows.append({
                "Model":    name,
                "Type":     mtype,
                "Term":     term,
                "N_Tokens": n,
                "Subwords": " | ".join(subs),
                "Has_UNK":  any(i == unk_id for i in ids),
            })

        print(f"vocab={vocab_sz:,}  fertility={fertility:.3f}  "
              f"probe_tok={avg_probe:.2f}  UNK={unk_rate:.4f}  "
              f"SNOMED_cov={cov:.3f}")

        return (
            {"Model": name, "Type": mtype,
             "Vocab Size":       vocab_sz,
             "Fertility":        round(fertility, 3),
             "UNK Rate":         round(unk_rate,  4),
             "Avg Tokens/Med":   round(avg_probe, 3),
             "Avg Tokens/Plain": round(avg_plain, 3),
             "Ratio Med/Plain":  round(avg_probe / max(avg_plain, 0.001), 3),
             "SNOMED Coverage":  cov},
            pd.DataFrame(term_rows),
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return {}, pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — KEY-TERM EXTRACTION
#  SNOMED-boosted: terms present in SNOMED_VOCAB are returned
#  first and appear twice (giving them double matching weight)
# ═══════════════════════════════════════════════════════════════

def extract_key_terms(query: str, min_len: int = 3) -> list[str]:
    """
    1. Strip punctuation, lowercase, remove STOPWORDS.
    2. SNOMED-boost: terms found in the SNOMED vocabulary are
       prepended so they appear twice → higher chance of hitting
       relevance threshold in short top-k lists.
    3. Falls back to any word ≥ min_len if everything was a stopword.

    min_len=3 preserves clinical abbreviations: DVT, EEA, BMI, MRI…
    """
    words = re.sub(r"[^\w\s]", " ", query).lower().split()
    terms = [w for w in words if len(w) >= min_len and w not in STOPWORDS]
    if not terms:
        terms = [w for w in words if len(w) >= min_len]

    # SNOMED boost
    snomed_terms = [t for t in terms if t in SNOMED_VOCAB]
    return snomed_terms + terms   # snomed terms appear first AND are duplicated


# ═══════════════════════════════════════════════════════════════
#  SECTION 9 — RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(relevances: list[int], sims: np.ndarray) -> dict:
    k = len(relevances)
    hits = sum(relevances)
    prec = hits / k
    recall = min(hits / max(k, 1), 1.0)
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevances) if r)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, hits)))
    ndcg = dcg / idcg if idcg else 0.0
    ranks = [1 / (i + 1) for i, r in enumerate(relevances) if r]
    mrr = float(np.mean(ranks)) if ranks else 0.0
    avg_s = float(np.mean(sims))
    return {
        "Prec@5":   round(prec,   3),
        "Recall@5": round(recall, 3),
        "Avg_Sim":  round(avg_s,  4),
        "MRR@5":    round(mrr,    3),
        "nDCG@5":   round(ndcg,   3),
        "Quality":  round((prec + avg_s + mrr + ndcg) / 4, 4),
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 10 — SINGLE-MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_model(
    name:        str,
    hf_path:     str,
    mtype:       str,
    size_tier:   str,
    chunks:      list[str],
    chunk_label: str = "sentence",
) -> tuple[dict | None, pd.DataFrame | None]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = BATCH_SIZES[size_tier]
    t0 = time.time()
    proc = psutil.Process()

    print(f"\n  {'─'*62}")
    print(f"  {name}  [{mtype}]  tier={size_tier}  "
          f"batch={batch}  device={device}  fp16={USE_FP16 and device=='cuda'}")

    try:
        model = SentenceTransformer(hf_path, device=device)
        if device == "cuda" and USE_FP16:
            model.half()

        embeddings = model.encode(
            chunks,
            batch_size=batch,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embed_time = time.time() - t0
        ram_mb = proc.memory_info().rss / 1e6
        vram_mb = torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0.0

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # cosine sim for unit vectors
        index.add(embeddings.astype(np.float32))

        rows = []
        for qi, (q, qlbl) in enumerate(
            tqdm(list(zip(QUERIES, QUERY_LABELS)),
                 desc="  queries", ncols=70, leave=False)
        ):
            key_terms = extract_key_terms(q)
            q_emb = model.encode([q], normalize_embeddings=True)[
                0].astype(np.float32)
            D, I = index.search(q_emb.reshape(1, -1), 5)
            top = [chunks[i] for i in I[0]]
            # relevance: a chunk is relevant if any key-term appears in it
            relevances = [
                1 if any(t in c.lower() for t in key_terms) else 0
                for c in top
            ]
            m = compute_metrics(relevances, D[0])
            rows.append({
                "QueryID":    qlbl,
                "Query":      q[:65] + "…" if len(q) > 65 else q,
                "Model":      name,
                "Type":       mtype,
                "ChunkStyle": chunk_label,
                **m,
            })

        df = pd.DataFrame(rows)
        avg = df.mean(numeric_only=True)
        total = time.time() - t0

        summary = {
            "Model":           name,
            "Type":            mtype,
            "ChunkStyle":      chunk_label,
            "SizeTier":        size_tier,
            "Dim":             dim,
            "Embed Time (s)":  round(embed_time, 1),
            "Total Time (s)":  round(total,      1),
            "RAM (MB)":        round(ram_mb,      0),
            "VRAM (MB)":       round(vram_mb,     0),
            "Avg Prec@5":      round(avg["Prec@5"],   3),
            "Avg Recall@5":    round(avg["Recall@5"], 3),
            "Avg Sim@5":       round(avg["Avg_Sim"],  4),
            "Avg MRR@5":       round(avg["MRR@5"],    3),
            "Avg nDCG@5":      round(avg["nDCG@5"],   3),
            "Overall Quality": round(avg["Quality"],  4),
        }

        pd.DataFrame([summary]).to_csv(
            os.path.join(OUT_DIR, f"ckpt_{name}_{chunk_label}.csv"), index=False
        )
        print(f"  → Quality={summary['Overall Quality']:.4f}  "
              f"Prec={summary['Avg Prec@5']:.3f}  "
              f"nDCG={summary['Avg nDCG@5']:.3f}  "
              f"time={embed_time:.1f}s  VRAM={vram_mb:.0f} MB")
        return summary, df

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    finally:
        # Always free GPU before the next model loads
        try:
            del model, embeddings, index
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()


# ═══════════════════════════════════════════════════════════════
#  SECTION 11 — FIGURE UTILITIES
# ═══════════════════════════════════════════════════════════════

def savefig(fig: plt.Figure, fname: str):
    path = os.path.join(OUT_DIR, f"{fname}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  {fname}.png")


def leg():
    return [mpatches.Patch(color=C_GEN, label="General-purpose"),
            mpatches.Patch(color=C_MED, label="Medical domain-adapted")]


def mcols(df: pd.DataFrame) -> list[str]:
    return [C_MAP.get(t, "#888") for t in df["Type"]]


# ═══════════════════════════════════════════════════════════════
#  SECTION 12 — 13 FIGURES
# ═══════════════════════════════════════════════════════════════

# ── Fig 01 : Dataset overview ────────────────────────────────────
def fig01_dataset_overview(corpus: pd.DataFrame):
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle("Fig 01 — Dataset Overview",
                 fontsize=15, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A: mtsamples specialty distribution (top 15)
    ax = fig.add_subplot(gs[0, :2])
    top15 = corpus.loc[corpus["source"] == "mtsamples", "specialty"]\
        .value_counts().head(15)
    bars = ax.barh(top15.index[::-1], top15.values[::-1],
                   color=SPEC_P[:len(top15)][::-1],
                   edgecolor="white", linewidth=0.4, height=0.72)
    for bar in bars:
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width())}", va="center", fontsize=8.5)
    ax.set_xlabel("Document count")
    ax.set_title("A   mtsamples — specialty distribution (top 15)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", linewidth=0.4)

    # B: X.csv class distribution (from classes.txt)
    ax2 = fig.add_subplot(gs[0, 2])
    xcls = corpus.loc[corpus["source"] ==
                      "x_corpus", "class_label"].value_counts()
    class_colors = [C_GEN, C_MED, "#3BAD7A", "#F2A623"][:len(xcls)]
    bars2 = ax2.bar(xcls.index, xcls.values,
                    color=class_colors, edgecolor="white", linewidth=0.5)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{int(bar.get_height())}", ha="center", fontsize=9)
    ax2.set_xlabel("")
    ax2.set_title("B   X.csv class distribution\n(from classes.txt)",
                  fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=9)

    # C: word-count histogram by source
    ax3 = fig.add_subplot(gs[1, :2])
    for src, color in [("mtsamples", C_GEN), ("x_corpus", C_MED)]:
        sub = corpus.loc[corpus["source"] == src, "word_count"]
        if len(sub):
            ax3.hist(sub, bins=60, color=color, alpha=0.65,
                     label=src, edgecolor="none")
    ax3.set_xlabel("Word count per document")
    ax3.set_ylabel("Frequency")
    ax3.set_title("C   Document length distribution",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # D: word-count boxplot by X.csv class
    ax4 = fig.add_subplot(gs[1, 2])
    xsub = corpus[corpus["source"] == "x_corpus"].copy()
    if len(xsub):
        order = (xsub.groupby("class_label")["word_count"]
                 .median().sort_values(ascending=False).index)
        sns.boxplot(data=xsub, y="class_label", x="word_count",
                    order=order, palette="Blues_d",
                    linewidth=0.7, fliersize=2, ax=ax4)
    ax4.set_xlabel("Word count")
    ax4.set_ylabel("")
    ax4.set_title("D   Length by X.csv class", fontsize=11, fontweight="bold")
    ax4.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    savefig(fig, "fig01_dataset_overview")


# ── Fig 02 : Chunking strategy comparison ────────────────────────
def fig02_chunking(s_meta: pd.DataFrame, f_meta: pd.DataFrame,
                   chunk_sum_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    fig.suptitle("Fig 02 — Chunking Strategy Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    ax = axes[0, 0]
    ax.hist(s_meta["word_count"], bins=50, color=C_GEN, alpha=0.7,
            label="Sentence", edgecolor="none")
    ax.hist(f_meta["word_count"], bins=50, color=C_MED, alpha=0.6,
            label="Fixed-window", edgecolor="none")
    ax.set_xlabel("Words per chunk")
    ax.set_ylabel("Frequency")
    ax.set_title("A   Chunk length distribution",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[0, 1]
    sc = s_meta.groupby("specialty").size(
    ).sort_values(ascending=False).head(12)
    ax2.barh(sc.index[::-1], sc.values[::-1], color=C_GEN,
             edgecolor="white", linewidth=0.4, height=0.72)
    ax2.set_xlabel("Chunk count")
    ax2.set_title("B   Sentence chunks per specialty (top 12)",
                  fontsize=11, fontweight="bold")

    ax3 = axes[1, 0]
    comb = pd.concat([s_meta[["word_count"]].assign(style="Sentence"),
                      f_meta[["word_count"]].assign(style="Fixed-window")])
    sns.violinplot(data=comb, x="style", y="word_count",
                   palette={"Sentence": C_GEN, "Fixed-window": C_MED},
                   inner="quartile", linewidth=0.8, ax=ax3)
    ax3.set_xlabel("")
    ax3.set_ylabel("Words per chunk")
    ax3.set_title("C   Chunk length violin", fontsize=11, fontweight="bold")

    ax4 = axes[1, 1]
    if not chunk_sum_df.empty and len(chunk_sum_df) >= 2:
        mets = ["Avg Prec@5", "Avg Recall@5", "Avg MRR@5", "Avg nDCG@5"]
        x = np.arange(len(mets))
        w = 0.35
        for i, (row, color) in enumerate(
                zip(chunk_sum_df.itertuples(), [C_GEN, C_MED])):
            style = getattr(row, "ChunkStyle", f"s{i}")
            # safe attribute access: spaces → underscores, @ → _
            vals = []
            for m in mets:
                safe = m.replace(" ", "_").replace("@", "_")
                vals.append(getattr(row, safe, 0))
            bars = ax4.bar(x + i*w - w/2, vals, w, label=style,
                           color=color, edgecolor="white")
            for bar in bars:
                ax4.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.005,
                         f"{bar.get_height():.3f}",
                         ha="center", fontsize=7.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(["Prec@5", "Recall@5", "MRR@5", "nDCG@5"])
        ax4.set_ylim(0, 1.05)
        ax4.set_title("D   Retrieval metrics by chunk style (MiniLM-L6)",
                      fontsize=11, fontweight="bold")
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, "Chunking comparison\nnot available",
                 ha="center", va="center", transform=ax4.transAxes)

    fig.tight_layout()
    savefig(fig, "fig02_chunking_comparison")


# ── Fig 03 : Tokenizer overview ───────────────────────────────────
def fig03_tokenizer_overview(tok_df: pd.DataFrame):
    if tok_df.empty:
        print("  Skipping fig03 — no tokenizer data")
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
    fig.suptitle("Fig 03 — Tokenizer Analysis: Vocabulary Adaptation",
                 fontsize=14, fontweight="bold", y=1.01)
    colors = mcols(tok_df)

    ax = axes[0, 0]
    bars = ax.barh(tok_df["Model"], tok_df["Fertility"],
                   color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax.axvline(1.0, color="#999", ls="--", lw=0.9, label="1 tok/word")
    for bar in bars:
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Avg subwords per word (SNOMED probe + plain mix)")
    ax.set_title("A   Token fertility", fontsize=11, fontweight="bold")
    ax.legend(handles=leg()+[mpatches.Patch(color="#999", label="ideal")],
              fontsize=7)

    ax2 = axes[0, 1]
    x = np.arange(len(tok_df))
    w = 0.36
    ax2.bar(x-w/2, tok_df["Avg Tokens/Med"], w, color=colors,
            edgecolor="white", linewidth=0.4, label="SNOMED probe terms")
    ax2.bar(x+w/2, tok_df["Avg Tokens/Plain"], w,
            color="lightsteelblue", edgecolor="white",
            linewidth=0.4, alpha=0.8, label="Plain English")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tok_df["Model"], rotation=32, ha="right", fontsize=8)
    ax2.set_ylabel("Avg tokens per term")
    ax2.set_title("B   SNOMED probe terms vs plain English",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    ax3 = axes[1, 0]
    ax3.barh(tok_df["Model"], tok_df["Ratio Med/Plain"],
             color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax3.axvline(1.0, color="#999", ls="--", lw=0.9)
    for i, val in enumerate(tok_df["Ratio Med/Plain"]):
        ax3.text(val+0.01, i, f"{val:.2f}", va="center", fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Ratio  (1.0 = equal tokenization efficiency)")
    ax3.set_title("C   Tokenization ratio: SNOMED ÷ plain",
                  fontsize=11, fontweight="bold")
    ax3.legend(handles=leg(), fontsize=8)

    ax4 = axes[1, 1]
    ax4.barh(tok_df["Model"], tok_df["Vocab Size"]/1000,
             color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax4.invert_yaxis()
    ax4.set_xlabel("Vocabulary size (thousands)")
    ax4.set_title("D   Tokenizer vocabulary size",
                  fontsize=11, fontweight="bold")
    ax4.legend(handles=leg(), fontsize=8)
    for i, val in enumerate(tok_df["Vocab Size"]):
        ax4.text(val/1000+0.5, i, f"{val:,}", va="center", fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig03_tokenizer_overview")


# ── Fig 04 : SNOMED probe term heatmap ───────────────────────────
def fig04_term_heatmap(tok_term_df: pd.DataFrame):
    if tok_term_df.empty:
        return
    pivot = tok_term_df.pivot_table(
        index="Term", columns="Model", values="N_Tokens", aggfunc="mean"
    )
    pivot = pivot[[c for c in MODEL_ORDER if c in pivot.columns]]
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 14), facecolor=BG)
    fig.suptitle(
        f"Fig 04 — SNOMED Probe Term Heatmap  (n={len(pivot)} terms)\n"
        "Darker = more subwords = worse medical vocabulary coverage",
        fontsize=13, fontweight="bold")

    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.3, linecolor=BG_AX,
                annot_kws={"size": 7.5},
                cbar_kws={"shrink": 0.6, "label": "Tokens per term"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                       ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.5)
    ax.set_xlabel("")
    ax.set_ylabel("SNOMED term (sampled from vocab.txt)")

    for label in ax.get_xticklabels():
        row = tok_term_df.loc[tok_term_df["Model"]
                              == label.get_text(), "Type"].values
        if len(row):
            label.set_color(C_MAP.get(row[0], TEXT))

    fig.tight_layout()
    savefig(fig, "fig04_snomed_term_heatmap")


# ── Fig 05 : Quality ranking ─────────────────────────────────────
def fig05_quality_ranking(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Fig 05 — Retrieval Quality Overview",
                 fontsize=14, fontweight="bold", y=1.01)

    df = summary_df.sort_values("Overall Quality", ascending=True)
    col = mcols(df)
    ax = axes[0]
    bars = ax.barh(df["Model"], df["Overall Quality"],
                   color=col, edgecolor="white", linewidth=0.5, height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w+0.003, bar.get_y()+bar.get_height()/2,
                f"{w:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, df["Overall Quality"].max()*1.20)
    ax.set_xlabel("Overall Quality")
    ax.set_title("A   Ranked by overall quality",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=leg(), fontsize=9, loc="lower right")

    ax2 = axes[1]
    df2 = summary_df.sort_values("Overall Quality", ascending=False)
    mets = ["Avg Prec@5", "Avg MRR@5", "Avg nDCG@5", "Avg Recall@5"]
    met_c = ["#4C8EDA", "#E8593C", "#3BAD7A", "#F2A623"]
    bot = np.zeros(len(df2))
    for m, c in zip(mets, met_c):
        v = df2[m].values
        ax2.barh(df2["Model"], v, left=bot, color=c,
                 edgecolor="white", linewidth=0.3, height=0.65, label=m)
        bot += v
    ax2.invert_yaxis()
    ax2.set_xlabel("Stacked metric contribution")
    ax2.set_title("B   Metric breakdown per model",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    savefig(fig, "fig05_quality_ranking")


# ── Fig 06 : Model × metric heatmap ──────────────────────────────
def fig06_metric_heatmap(summary_df: pd.DataFrame):
    heat = summary_df.set_index("Model")[METRIC_COLS]
    heat = heat.reindex([m for m in MODEL_ORDER if m in heat.index])

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    fig.suptitle("Fig 06 — Model × Metric Heatmap",
                 fontsize=14, fontweight="bold")
    sns.heatmap(heat, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8, "label": "Score"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22,
                       ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    for label in ax.get_yticklabels():
        row = summary_df.loc[summary_df["Model"]
                             == label.get_text(), "Type"].values
        if len(row):
            label.set_color(C_MAP.get(row[0], TEXT))
    ax.set_title("Y-axis colour = model family", fontsize=9, color=TEXT_S)
    fig.tight_layout()
    savefig(fig, "fig06_metric_heatmap")


# ── Fig 07 : Per-query heatmap ───────────────────────────────────
def fig07_per_query_heatmap(per_df: pd.DataFrame):
    if per_df.empty:
        return
    pivot = per_df.pivot_table(index="Model", columns="QueryID",
                               values="Quality", aggfunc="mean")
    pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index])

    fig, ax = plt.subplots(figsize=(22, 7), facecolor=BG)
    fig.suptitle("Fig 07 — Per-Query Quality Heatmap  (model × query)",
                 fontsize=14, fontweight="bold")
    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                linewidths=0.3, linecolor=BG,
                annot_kws={"size": 7.5},
                cbar_kws={"shrink": 0.5, "label": "Quality"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    for label in ax.get_yticklabels():
        row = per_df.loc[per_df["Model"] == label.get_text(), "Type"].values
        if len(row):
            label.set_color(C_MAP.get(row[0], TEXT))
    ax.set_xlabel("Query ID  (Q01–Q25)")
    ax.set_ylabel("")
    fig.tight_layout()
    savefig(fig, "fig07_per_query_heatmap")


# ── Fig 08 : Quality distribution violin ─────────────────────────
def fig08_violin(per_df: pd.DataFrame):
    if per_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Fig 08 — Per-Query Quality Distribution",
                 fontsize=14, fontweight="bold", y=1.01)

    order = per_df.groupby("Model")["Quality"].median(
    ).sort_values(ascending=False).index
    palette = {m: C_MAP.get(
        per_df.loc[per_df["Model"] == m, "Type"].iloc[0], "#888")
        for m in per_df["Model"].unique()}

    sns.violinplot(data=per_df, x="Model", y="Quality",
                   order=order, palette=palette,
                   inner="quartile", linewidth=0.8, ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(),
                            rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("Quality score (per query)")
    axes[0].set_title("A   Per-model distribution",
                      fontsize=11, fontweight="bold")
    axes[0].legend(handles=leg(), fontsize=8)
    axes[0].set_xlabel("")

    sns.violinplot(data=per_df, x="Type", y="Quality",
                   palette={"general": C_GEN, "medical": C_MED},
                   inner="box", linewidth=0.8, ax=axes[1])
    sns.stripplot(data=per_df, x="Type", y="Quality",
                  palette={"general": C_GEN, "medical": C_MED},
                  alpha=0.15, size=3, jitter=True, ax=axes[1])
    axes[1].set_xticklabels(["General-purpose", "Medical domain-adapted"],
                            fontsize=11)
    axes[1].set_ylabel("Quality score")
    axes[1].set_title("B   By model family", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("")

    fig.tight_layout()
    savefig(fig, "fig08_violin_distribution")


# ── Fig 09 : Efficiency scatter ───────────────────────────────────
def fig09_efficiency(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Fig 09 — Efficiency Analysis",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, xcol, title, xlabel in [
        (axes[0], "Embed Time (s)",
         "A   Efficiency frontier", "Embedding time (s)"),
        (axes[1], "VRAM (MB)",      "B   VRAM cost vs quality", "Peak VRAM (MB)"),
    ]:
        for _, row in summary_df.iterrows():
            c = C_MAP.get(row["Type"], "#888")
            ax.scatter(row[xcol], row["Overall Quality"],
                       s=row["Dim"]/3, color=c, alpha=0.85,
                       edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(row["Model"],
                        (row[xcol], row["Overall Quality"]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=8.5, color=TEXT)
        ax.axvline(summary_df[xcol].median(), color=GRID, ls="--", lw=1)
        ax.axhline(summary_df["Overall Quality"].median(),
                   color=GRID, ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Overall Quality")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(handles=leg(), fontsize=9)
        note = ax.get_title()
        ax.set_title(note + "\n(bubble size ∝ embedding dimension)",
                     fontsize=10, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig09_efficiency")


# ── Fig 10 : General vs Medical head-to-head ─────────────────────
def fig10_group_comparison(group_df: pd.DataFrame):
    if group_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("Fig 10 — General vs Medical Domain-Adapted: Head-to-Head",
                 fontsize=14, fontweight="bold", y=1.01)

    mets = ["Avg Prec@5", "Avg Recall@5", "Avg MRR@5", "Avg nDCG@5"]
    x = np.arange(len(mets))
    w = 0.35
    ax = axes[0]
    for i, (gname, color) in enumerate({"general": C_GEN, "medical": C_MED}.items()):
        if gname in group_df.index:
            vals = [group_df.loc[gname, m] for m in mets]
            bars = ax.bar(x+i*w-w/2, vals, w, label=gname,
                          color=color, edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.007,
                        f"{bar.get_height():.3f}",
                        ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Prec@5", "Recall@5", "MRR@5", "nDCG@5"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("A   Mean scores per metric", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    all_mets = ["Avg Prec@5", "Avg Recall@5",
                "Avg MRR@5", "Avg nDCG@5", "Overall Quality"]
    xi = range(len(all_mets))
    for gname, color in {"general": C_GEN, "medical": C_MED}.items():
        if gname in group_df.index:
            vals = [group_df.loc[gname, m] for m in all_mets]
            ax2.plot(xi, vals, color=color, lw=2.5, marker="o",
                     markersize=8, label=gname.capitalize())
            for x_, y_ in zip(xi, vals):
                ax2.annotate(f"{y_:.3f}", (x_, y_),
                             textcoords="offset points", xytext=(0, 9),
                             ha="center", fontsize=9, color=color)
    ax2.set_xticks(xi)
    ax2.set_xticklabels(["Prec@5", "Recall@5", "MRR@5", "nDCG@5", "Overall"],
                        fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("B   Parallel coordinates", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", lw=0.5)

    fig.tight_layout()
    savefig(fig, "fig10_group_comparison")


# ── Fig 11 : Metric correlation matrix ───────────────────────────
def fig11_correlation(per_df: pd.DataFrame):
    if per_df.empty:
        return
    cols = ["Prec@5", "Recall@5", "Avg_Sim", "MRR@5", "nDCG@5", "Quality"]
    corr = per_df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
    fig.suptitle("Fig 11 — Retrieval Metric Correlation Matrix\n"
                 "(per-query scores across all models)",
                 fontsize=13, fontweight="bold")
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm",
                annot=True, fmt=".2f", vmin=-1, vmax=1,
                linewidths=0.5, linecolor=BG_AX,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                       ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig11_metric_correlation")


# ── Fig 12 : X.csv class-stratified retrieval ────────────────────
def fig12_class_retrieval(per_df: pd.DataFrame):
    """
    Uses QUERY_CLASS (aligned with classes.txt) to stratify
    per-query scores by clinical class: Surgery, Medical Records,
    Internal Medicine, Other.
    """
    if per_df.empty:
        return

    q_to_cls: dict[str, str] = {}
    for cls, qids in QUERY_CLASS.items():
        for q in qids:
            q_to_cls[q] = cls

    per2 = per_df.copy()
    per2["QueryClass"] = per2["QueryID"].map(q_to_cls).fillna("Other")
    classes = list(QUERY_CLASS.keys())

    grp = per2.groupby(["QueryClass", "Type"])["Quality"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Fig 12 — Class-Stratified Retrieval Quality\n"
                 "(classes from classes.txt: Surgery / Medical Records / "
                 "Internal Medicine / Other)",
                 fontsize=13, fontweight="bold", y=1.02)

    ax = axes[0]
    x = np.arange(len(classes))
    w = 0.35
    for i, (gname, color) in enumerate({"general": C_GEN, "medical": C_MED}.items()):
        sub = grp[grp["Type"] == gname]
        vals = []
        for cls in classes:
            hit = sub.loc[sub["QueryClass"] == cls, "Quality"].values
            vals.append(float(hit[0]) if len(hit) else 0.0)
        ax.bar(x+i*w-w/2, vals, w, label=gname,
               color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Quality score")
    ax.set_title("A   Quality by clinical class",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    grp2 = per2.groupby(["Model", "QueryClass"])[
        "Quality"].mean().reset_index()
    piv = grp2.pivot(index="Model", columns="QueryClass", values="Quality")
    piv = piv.reindex(index=[m for m in MODEL_ORDER if m in piv.index])
    sns.heatmap(piv, ax=ax2, cmap="YlOrRd", annot=True, fmt=".3f",
                linewidths=0.4, linecolor=BG,
                annot_kws={"size": 9.5},
                cbar_kws={"shrink": 0.7})
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20,
                        ha="right", fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
    ax2.set_title("B   Model × class heatmap", fontsize=11, fontweight="bold")
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    for label in ax2.get_yticklabels():
        row = per_df.loc[per_df["Model"] == label.get_text(), "Type"].values
        if len(row):
            label.set_color(C_MAP.get(row[0], TEXT))

    fig.tight_layout()
    savefig(fig, "fig12_class_retrieval")


# ── Fig 13 : SNOMED vocabulary coverage ──────────────────────────
def fig13_snomed_coverage(tok_df: pd.DataFrame):
    """
    SNOMED coverage = fraction of 5,000 sampled SNOMED tokens
    that each model tokenizer encodes as exactly 1 subword.
    Higher coverage → the model's vocabulary already contains
    these medical concepts as whole units rather than fragments.

    This directly measures the effect of domain vocabulary adaptation.
    """
    if tok_df.empty or "SNOMED Coverage" not in tok_df.columns:
        print("  Skipping fig13 — no SNOMED coverage data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor=BG)
    fig.suptitle(
        "Fig 13 — SNOMED Vocabulary Coverage\n"
        "Fraction of SNOMED tokens encoded as a single subword\n"
        "(source: vocab.txt — SNOMED-derived NLP vocabulary)",
        fontsize=13, fontweight="bold", y=1.03,
    )

    # A: coverage bar chart
    ax = axes[0]
    df = tok_df.sort_values("SNOMED Coverage", ascending=True)
    cols = mcols(df)
    bars = ax.barh(df["Model"], df["SNOMED Coverage"],
                   color=cols, edgecolor="white", linewidth=0.5, height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w+0.003, bar.get_y()+bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(1.0, df["SNOMED Coverage"].max()*1.25))
    ax.set_xlabel("Single-subword coverage rate  (higher = better)")
    ax.set_title("A   SNOMED coverage per model",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=leg(), fontsize=9)

    # B: coverage vs fertility scatter
    ax2 = axes[1]
    for _, row in tok_df.iterrows():
        c = C_MAP.get(row["Type"], "#888")
        ax2.scatter(row["SNOMED Coverage"], row["Fertility"],
                    s=180, color=c, alpha=0.85,
                    edgecolors="white", linewidths=0.8, zorder=3)
        ax2.annotate(row["Model"],
                     (row["SNOMED Coverage"], row["Fertility"]),
                     textcoords="offset points", xytext=(7, 3),
                     fontsize=8.5, color=TEXT)

    # Ideal quadrant: high coverage, low fertility
    cov_med = tok_df["SNOMED Coverage"].median()
    fer_med = tok_df["Fertility"].median()
    ax2.axvline(cov_med, color=GRID, ls="--", lw=1,
                label=f"Median coverage ({cov_med:.3f})")
    ax2.axhline(fer_med, color=GRID, ls=":", lw=1,
                label=f"Median fertility ({fer_med:.3f})")
    ax2.set_xlabel("SNOMED coverage  (fraction of vocab encoded as 1 token)")
    ax2.set_ylabel("Token fertility  (avg subwords per word)")
    ax2.set_title("B   Coverage vs fertility\n"
                  "(ideal: top-right = high coverage, lower fertility)",
                  fontsize=11, fontweight="bold")
    ax2.legend(handles=leg()+[
        mpatches.Patch(color=GRID, label=f"Median coverage")
    ], fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig13_snomed_coverage")


def make_all_plots(summary_df, per_df, tok_df, tok_term_df,
                   corpus, s_meta, f_meta, chunk_sum_df, group_df):
    print("\n── Generating 13 figures ─────────────────────────────────")
    fig01_dataset_overview(corpus)
    fig02_chunking(s_meta, f_meta, chunk_sum_df)
    fig03_tokenizer_overview(tok_df)
    fig04_term_heatmap(tok_term_df)
    fig05_quality_ranking(summary_df)
    fig06_metric_heatmap(summary_df)
    fig07_per_query_heatmap(per_df)
    fig08_violin(per_df)
    fig09_efficiency(summary_df)
    fig10_group_comparison(group_df)
    fig11_correlation(per_df)
    fig12_class_retrieval(per_df)
    fig13_snomed_coverage(tok_df)


# ═══════════════════════════════════════════════════════════════
#  SECTION 13 — MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global STOPWORDS, X_LABEL_MAP, SNOMED_VOCAB, SNOMED_PROBE

    t_wall = time.time()

    print("\n" + "═"*66)
    print("  Medical RAG Evaluation  —  RTX 5070 Ti  (Blackwell/GB203)")
    print("═"*66)
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {props.total_memory/1e9:.1f} GB")
        print(f"  SMs     : {props.multi_processor_count}")
    print(f"  fp16    : {USE_FP16}\n")

    # ── load support files ───────────────────────────────────────
    print("── Loading support files ─────────────────────────────────")
    STOPWORDS = load_stopwords(SW_FILE)
    X_LABEL_MAP = load_classes(CLS_FILE)

    # vocab.txt — SNOMED NLP vocabulary
    SNOMED_VOCAB, SNOMED_PROBE = load_snomed_vocab(VOC_FILE, STOPWORDS)

    # ── corpus ───────────────────────────────────────────────────
    print("\n── Building corpus ───────────────────────────────────────")
    corpus = build_corpus(MAX_DOCS)
    corpus.to_csv(os.path.join(OUT_DIR, "corpus.csv"), index=False)
    print(f"  corpus.csv saved  ({len(corpus):,} rows)")

    # ── chunking ─────────────────────────────────────────────────
    print("\n── Chunking ──────────────────────────────────────────────")
    s_chunks, f_chunks, s_meta, f_meta = build_chunks(corpus)

    # ── phase 1 : tokenizer analysis ─────────────────────────────
    print("\n── Phase 1 : Tokenizer Analysis ──────────────────────────")
    print(f"  SNOMED probe set : {len(SNOMED_PROBE)} terms from vocab.txt")
    tok_rows, tok_frames = [], []
    for name, (path, mtype, _) in MODELS.items():
        summary, tdf = analyze_tokenizer(name, path, mtype)
        if summary:
            tok_rows.append(summary)
            tok_frames.append(tdf)

    tok_df = pd.DataFrame(tok_rows)
    tok_term_df = (pd.concat(tok_frames, ignore_index=True)
                   if tok_frames else pd.DataFrame())
    tok_df.to_csv(os.path.join(OUT_DIR, "tokenizer_summary.csv"),  index=False)
    tok_term_df.to_csv(os.path.join(
        OUT_DIR, "tokenizer_per_term.csv"), index=False)

    if not tok_df.empty:
        print("\n  SNOMED Coverage Summary:")
        for _, row in tok_df.sort_values("SNOMED Coverage",
                                         ascending=False).iterrows():
            bar = "█" * int(row["SNOMED Coverage"] * 30)
            print(
                f"    {row['Model']:<16s} {bar:<30s} {row['SNOMED Coverage']:.3f}")

    # ── phase 2 : chunking comparison ────────────────────────────
    print("\n── Phase 2 : Chunking Comparison (MiniLM-L6) ────────────")
    pname, (ppath, pmtype, ptier) = "MiniLM-L6", MODELS["MiniLM-L6"]
    chunk_rows = []
    for label, chunks in [("sentence", s_chunks), ("fixed-window", f_chunks)]:
        s, _ = evaluate_model(pname, ppath, pmtype, ptier, chunks, label)
        if s:
            chunk_rows.append(s)
    chunk_sum_df = pd.DataFrame(chunk_rows)
    chunk_sum_df.to_csv(os.path.join(
        OUT_DIR, "chunking_comparison.csv"), index=False)

    # ── phase 3 : full retrieval evaluation ──────────────────────
    print("\n── Phase 3 : Retrieval Evaluation — all 9 models ─────────")
    summary_rows, per_frames = [], []
    for name, (path, mtype, tier) in MODELS.items():
        summary, per_df = evaluate_model(name, path, mtype, tier,
                                         s_chunks, "sentence")
        if summary:
            summary_rows.append(summary)
        if per_df is not None:
            per_frames.append(per_df)

    summary_df = (pd.DataFrame(summary_rows)
                  .sort_values("Overall Quality", ascending=False)
                  .reset_index(drop=True))
    per_df = (pd.concat(per_frames, ignore_index=True)
              if per_frames else pd.DataFrame())
    group_df = (summary_df.groupby("Type")[METRIC_COLS].mean().round(4)
                if not summary_df.empty else pd.DataFrame())

    summary_df.to_csv(os.path.join(
        OUT_DIR, "final_rag_comparison.csv"), index=False)
    per_df.to_csv(os.path.join(
        OUT_DIR, "per_query_results.csv"),    index=False)
    group_df.to_csv(os.path.join(OUT_DIR, "group_comparison.csv"))

    print("\n── Final Summary ──────────────────────────────────────────")
    print(summary_df[["Model", "Type", "Overall Quality", "Avg Prec@5",
                      "Avg MRR@5", "Avg nDCG@5",
                      "Embed Time (s)", "VRAM (MB)"]].to_string(index=False))
    print("\n── Group Averages (General vs Medical) ────────────────────")
    print(group_df.to_string())

    # ── phase 4 : visualisations ──────────────────────────────────
    print("\n── Phase 4 : Visualisations ──────────────────────────────")
    make_all_plots(summary_df, per_df, tok_df, tok_term_df,
                   corpus, s_meta, f_meta, chunk_sum_df, group_df)

    # ── output manifest ───────────────────────────────────────────
    print("\n── Output Files ───────────────────────────────────────────")
    manifest = [
        ("corpus.csv",                   "Merged, deduplicated clinical corpus"),
        ("tokenizer_summary.csv",
         "Per-model: fertility/UNK/vocab/SNOMED coverage"),
        ("tokenizer_per_term.csv",
         f"Per-term subword breakdown ({len(SNOMED_PROBE)} SNOMED probe terms)"),
        ("chunking_comparison.csv",      "Sentence vs fixed-window metrics"),
        ("final_rag_comparison.csv",     "Full benchmark — 9 models"),
        ("per_query_results.csv",        "Per-query scores for all models"),
        ("group_comparison.csv",         "General vs medical group averages"),
        ("fig01_dataset_overview.png",
         "Dataset EDA (specialty dist, X classes, lengths)"),
        ("fig02_chunking_comparison.png", "Chunking strategy analysis"),
        ("fig03_tokenizer_overview.png",
         "Tokenizer fertility, SNOMED probe, vocab size"),
        ("fig04_snomed_term_heatmap.png", "SNOMED probe term fragmentation heatmap"),
        ("fig05_quality_ranking.png",    "Model quality ranking"),
        ("fig06_metric_heatmap.png",     "Model × metric heatmap"),
        ("fig07_per_query_heatmap.png",  "Model × query heatmap"),
        ("fig08_violin_distribution.png", "Quality distribution violins"),
        ("fig09_efficiency.png",         "Efficiency frontier (time & VRAM)"),
        ("fig10_group_comparison.png",   "General vs medical head-to-head"),
        ("fig11_metric_correlation.png", "Metric correlation matrix"),
        ("fig12_class_retrieval.png",
         "Retrieval by X.csv class (Surgery/MedRec/IntMed/Other)"),
        ("fig13_snomed_coverage.png",    "SNOMED full-vocab coverage per model"),
    ]
    for fname, desc in manifest:
        mark = "✓" if os.path.exists(os.path.join(OUT_DIR, fname)) else "✗"
        print(f"  {mark}  {fname:<46s}  {desc}")

    print(f"\n  Total wall time : {(time.time()-t_wall)/60:.1f} min")
    print(f"  Results in      : {os.path.abspath(OUT_DIR)}/\n")


if __name__ == "__main__":
    main()
