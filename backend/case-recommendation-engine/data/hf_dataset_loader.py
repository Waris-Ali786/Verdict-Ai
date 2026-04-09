import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm


# ──────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────

HF_DATASET_ID   = "Ibtehaj10/supreme-court-of-pak-judgments"
MAX_TEXT_LENGTH = 4000   # Trim judgments — some are 633k chars
CACHE_PATH      = "data/cache/sc_pakistan_cases.parquet"


# ──────────────────────────────────────────────────
# Case-type detection rules
# ──────────────────────────────────────────────────

CASE_TYPE_RULES = {
    "criminal": [
        "murder", "302 ppc", "section 302", "theft", "robbery", "assault",
        "criminal appeal", "fir", "accused", "prosecution", "narcotic",
        "cnsa", "peca", "terrorism", "kidnapping", "dacoity", "rape",
        "anti-terrorism", "death sentence", "life imprisonment",
    ],
    "family": [
        "divorce", "khul", "custody", "maintenance", "nikah", "marriage",
        "family court", "guardian", "minor child", "dower", "mehr",
        "muslim family laws", "dissolution of marriage",
    ],
    "civil": [
        "civil appeal", "breach of contract", "damages", "injunction",
        "property", "land", "specific performance", "tort", "negligence",
        "recovery", "plaintiff", "defendant", "suit for",
    ],
    "corporate": [
        "company", "secp", "director", "shareholder", "winding up",
        "merger", "securities", "arbitration", "commercial dispute",
    ],
    "constitutional": [
        "fundamental rights", "article 199", "article 184", "writ petition",
        "habeas corpus", "mandamus", "quo warranto", "constitutional petition",
    ],
    "service": [
        "civil servant", "government employee", "dismissal from service",
        "compulsory retirement", "service tribunal", "federal service",
        "provincial service", "promotion", "seniority", "pension",
    ],
    "tax": [
        "income tax", "sales tax", "customs duty", "fbr", "tax appeal",
        "tax tribunal", "commissioner inland revenue", "duty drawback",
    ],
}

OUTCOME_RULES = {
    "acquitted": [
        "appeal allowed", "allow the appeal", "conviction set aside",
        "acquitted", "sentence reduced", "bail granted",
        "order is set aside", "impugned judgment is set aside",
    ],
    "guilty": [
        "appeal dismissed", "dismiss the appeal", "conviction upheld",
        "sentence upheld", "dismissed", "upheld the conviction",
        "affirm the judgment",
    ],
    "settled": [
        "settlement", "compromise", "agreed terms", "consent decree",
    ],
    "pending": [
        "adjourned", "fixed for", "next date", "leave to appeal",
        "office objections",
    ],
}

RISK_RULES = {
    "high": [
        "murder", "terrorism", "death sentence", "capital punishment",
        "life imprisonment", "narcotics", "corruption", "kidnapping",
        "rape", "anti-terrorism", "scheduled offence",
    ],
    "low": [
        "adjourned", "leave refused", "tax", "service matter",
        "promotion", "seniority", "pension", "minor penalty",
    ],
}


# ──────────────────────────────────────────────────
# Detection helpers
# ──────────────────────────────────────────────────

def detect_case_type(text: str) -> str:
    t = text.lower()
    scores = {ct: sum(1 for kw in kws if kw in t)
              for ct, kws in CASE_TYPE_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "civil"


def detect_outcome(text: str) -> str:
    t = text.lower()
    for outcome, keywords in OUTCOME_RULES.items():
        if any(kw in t for kw in keywords):
            return outcome
    return "pending"


def detect_risk(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in RISK_RULES["high"]):
        return "high"
    if any(kw in t for kw in RISK_RULES["low"]):
        return "low"
    return "medium"


def extract_year(citation: str) -> int:
    """Extract year from citation like 'C.A.10_2021.pdf'."""
    match = re.search(r"(\d{4})", str(citation))
    if match:
        y = int(match.group(1))
        if 1947 <= y <= 2025:
            return y
    return 2020


def parse_citation(raw: any) -> str:
    """Parse citation from dict or string."""
    if isinstance(raw, dict):
        return raw.get("id", "Unknown")
    return str(raw)


def extract_statutes(text: str) -> str:
    """Extract statute references mentioned in text."""
    pattern = re.compile(
        r"(section|article|order|rule|act|ordinance)\s+\d+[\w\-]*"
        r"(?:\s+(?:of|ppc|crpc|peca|cnsa))?",
        re.IGNORECASE,
    )
    found = list({m.group(0).strip() for m in pattern.finditer(text)})
    return " | ".join(found[:8])  # max 8 statutes


def extract_title(citation: str, text: str) -> str:
    """Build a human-readable case title."""
    cit = citation.replace(".pdf", "").replace("_", " ")
    # Try to extract appellant vs respondent from text
    vs_match = re.search(
        r"([A-Z][A-Za-z\s\.]+(?:Ltd|Corp|Gov|Pvt)?)\s+[…\.]{0,3}\s*(?:Appellant|Petitioner)"
        r".*?([A-Z][A-Za-z\s\.]+)\s+[…\.]{0,3}\s*(?:Respondent|Defendant)",
        text[:1000],
    )
    if vs_match:
        appellant  = vs_match.group(1).strip()[:30]
        respondent = vs_match.group(2).strip()[:30]
        return f"{appellant} v. {respondent} — {cit}"
    return f"Supreme Court — {cit}"


# ──────────────────────────────────────────────────
# Main loader class
# ──────────────────────────────────────────────────

class HFDatasetLoader:
    """
    Downloads, processes, and caches the Supreme Court of Pakistan
    judgment dataset from HuggingFace.

    Samsung curriculum: OOP, Pandas, NumPy, real-world data pipelines.
    """

    def __init__(self, max_text_length: int = MAX_TEXT_LENGTH, use_cache: bool = True):
        self.max_text_length = max_text_length
        self.use_cache       = use_cache
        self._df: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────
    # Download from HuggingFace
    # ──────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Load dataset from cache or HuggingFace.
        Returns a clean Pandas DataFrame ready for all models.
        """
        if self._df is not None:
            return self._df

        # Try cache first
        if self.use_cache and os.path.exists(CACHE_PATH):
            print(f"[HF Loader] Loading from cache: {CACHE_PATH}")
            self._df = pd.read_parquet(CACHE_PATH)
            print(f"[HF Loader] Loaded {len(self._df)} cases from cache.")
            return self._df

        # Download from HuggingFace
        print(f"[HF Loader] Downloading from HuggingFace: {HF_DATASET_ID}")
        print("  This may take 1-2 minutes on first run...")

        try:
            from datasets import load_dataset
            raw = load_dataset(HF_DATASET_ID, split="train", trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download HuggingFace dataset: {e}\n"
                f"Make sure you have internet access and 'datasets' package installed.\n"
                f"Run: pip install datasets"
            )

        print(f"[HF Loader] Downloaded {len(raw)} cases. Processing...")
        self._df = self._process(raw)

        # Save cache
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        # Save without embeddings column (too large for parquet)
        cache_df = self._df.drop(columns=["hf_embedding"], errors="ignore")
        cache_df.to_parquet(CACHE_PATH, index=False)
        print(f"[HF Loader] Cached to {CACHE_PATH}")

        return self._df

    def _process(self, raw_dataset) -> pd.DataFrame:
        """
        Transform raw HuggingFace dataset into a clean DataFrame.

        Each row becomes:
          case_id, title, case_type, jurisdiction, facts,
          statutes, outcome, risk_level, year, court,
          citation, url, tags, full_text, hf_embedding
        """
        rows = []
        embeddings_list = []

        for i, row in enumerate(tqdm(raw_dataset, desc="  Processing judgments")):
            text     = str(row["text"])
            citation = parse_citation(row["citation_number"])
            emb      = row["embeddings"]  # list of 1024 floats

            # Trim text for model input
            trimmed_text = text[:self.max_text_length]

            case_type = detect_case_type(text)
            outcome   = detect_outcome(text[-2000:])   # check end of judgment
            risk      = detect_risk(text)
            year      = extract_year(citation)
            title     = extract_title(citation, text)
            statutes  = extract_statutes(text[:3000])

            rows.append({
                "case_id":      f"SC{i:04d}",
                "title":        title,
                "case_type":    case_type,
                "jurisdiction": "federal",
                "facts":        trimmed_text[:500],
                "statutes":     statutes,
                "outcome":      outcome,
                "risk_level":   risk,
                "year":         year,
                "court":        "Supreme Court of Pakistan",
                "citation":     citation.replace(".pdf", ""),
                "url":          f"https://supremecourt.gov.pk/judgments/{citation}",
                "tags":         f"{case_type} supreme court pakistan {year}",
                "full_text":    f"{title} {trimmed_text}",
            })
            embeddings_list.append(emb)

        df = pd.DataFrame(rows)
        # Attach embeddings as object column (numpy arrays)
        df["hf_embedding"] = embeddings_list
        return df

    # ──────────────────────────────────────────────
    # Embedding matrix
    # ──────────────────────────────────────────────

    def get_embeddings_matrix(self) -> np.ndarray:
        """
        Return pre-computed HF embeddings as a NumPy matrix.
        Shape: (1414, 1024)

        These are from mixedbread-ai/mxbai-embed-large-v1.
        L2-normalized for cosine similarity.
        """
        if self._embeddings is not None:
            return self._embeddings

        df = self.load()
        if "hf_embedding" not in df.columns:
            raise ValueError("hf_embedding column not found. Re-download without cache.")

        print("[HF Loader] Building embedding matrix (1414 × 1024)...")
        matrix = np.array(df["hf_embedding"].tolist(), dtype=np.float32)

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = matrix / norms

        print(f"[HF Loader] Embedding matrix shape: {self._embeddings.shape}")
        return self._embeddings

    # ──────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────

    def summary(self):
        """Print dataset statistics using Pandas."""
        df = self.load()
        print("\n" + "=" * 50)
        print("  Supreme Court of Pakistan — Dataset Summary")
        print("=" * 50)
        print(f"  Total cases:      {len(df)}")
        print(f"  Text avg length:  {df['full_text'].str.len().mean():.0f} chars")
        print(f"  Year range:       {df['year'].min()} – {df['year'].max()}")
        print(f"\n  Case types:")
        for ct, count in df["case_type"].value_counts().items():
            bar = "█" * (count // 10)
            print(f"    {ct:<16} {count:>4}  {bar}")
        print(f"\n  Outcomes:")
        for oc, count in df["outcome"].value_counts().items():
            print(f"    {oc:<16} {count:>4}")
        print(f"\n  Risk levels:")
        for rl, count in df["risk_level"].value_counts().items():
            print(f"    {rl:<16} {count:>4}")
        print("=" * 50)
