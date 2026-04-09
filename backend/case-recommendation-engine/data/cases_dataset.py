import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from data.hf_dataset_loader import HFDatasetLoader


# ──────────────────────────────────────────────────
# OOP: Legal resource data class (unchanged from v1)
# ──────────────────────────────────────────────────

@dataclass
class LegalResource:
    """A recommendable legal resource (statute, form, guideline)."""
    resource_id:   str
    title:         str
    resource_type: str
    description:   str
    url:           str
    tags:          List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────
# Legal resources knowledge base (expanded)
# ──────────────────────────────────────────────────

RESOURCES: List[LegalResource] = [
    LegalResource("R001", "Pakistan Penal Code 1860 — Full Text",
                  "statute", "Complete PPC covering all criminal offences and punishments.",
                  "https://pakistancode.gov.pk/english/ppc",
                  ["criminal", "PPC", "offence", "punishment"]),

    LegalResource("R002", "Code of Criminal Procedure 1898",
                  "statute", "Procedural law for criminal cases — FIR, bail, trial, appeals.",
                  "https://pakistancode.gov.pk/english/crpc",
                  ["criminal", "procedure", "CrPC", "bail", "FIR", "trial"]),

    LegalResource("R003", "Muslim Family Laws Ordinance 1961",
                  "statute", "Governs marriage, divorce, maintenance, and succession for Muslims.",
                  "https://pakistancode.gov.pk/english/mflo-1961",
                  ["family", "divorce", "marriage", "MFLO"]),

    LegalResource("R004", "West Pakistan Family Courts Act 1964",
                  "statute", "Establishes family courts and their jurisdiction over matrimonial disputes.",
                  "https://pakistancode.gov.pk/english/family-courts-act-1964",
                  ["family", "family court", "custody", "maintenance"]),

    LegalResource("R005", "Guardian and Wards Act 1890",
                  "statute", "Governs guardianship of minor children and their welfare.",
                  "https://pakistancode.gov.pk/english/guardian-wards-act-1890",
                  ["custody", "guardian", "minor", "children"]),

    LegalResource("R006", "Contract Act 1872 — Full Text",
                  "statute", "Foundation of Pakistani contract law — offer, acceptance, breach, remedies.",
                  "https://pakistancode.gov.pk/english/contract-act-1872",
                  ["contract", "breach", "commercial", "damages"]),

    LegalResource("R007", "Civil Procedure Code 1908",
                  "statute", "Procedure for civil suits — pleadings, evidence, appeals, execution.",
                  "https://pakistancode.gov.pk/english/cpc",
                  ["civil", "suit", "procedure", "CPC", "decree"]),

    LegalResource("R008", "Prevention of Electronic Crimes Act 2016",
                  "statute", "Pakistan's primary cybercrime legislation — digital offences and evidence.",
                  "https://pakistancode.gov.pk/english/peca-2016",
                  ["cybercrime", "PECA", "digital", "electronic"]),

    LegalResource("R009", "National Accountability Ordinance 1999",
                  "statute", "Establishes NAB and defines corruption offences and accountability.",
                  "https://nab.gov.pk/uploads/nab_ordinance.pdf",
                  ["NAB", "corruption", "accountability", "public official"]),

    LegalResource("R010", "Control of Narcotic Substances Act 1997",
                  "statute", "Defines narcotics offences, penalties, and enforcement mechanisms.",
                  "https://pakistancode.gov.pk/english/cnsa-1997",
                  ["narcotics", "drugs", "CNSA", "trafficking"]),

    LegalResource("R011", "Land Acquisition Act 1894",
                  "statute", "Governs compulsory acquisition of private land and compensation.",
                  "https://pakistancode.gov.pk/english/land-acquisition-act-1894",
                  ["land", "property", "acquisition", "compensation"]),

    LegalResource("R012", "Income Tax Ordinance 2001",
                  "statute", "Primary income tax legislation — assessment, appeals, penalties.",
                  "https://fbr.gov.pk/income-tax-ordinance-2001",
                  ["tax", "income tax", "FBR", "assessment"]),

    LegalResource("R013", "Sindh Civil Servants Act 1973",
                  "statute", "Governs appointment, conduct, and discipline of Sindh civil servants.",
                  "https://pakistancode.gov.pk/sindh/civil-servants-act-1973",
                  ["civil service", "government employee", "service", "sindh"]),

    LegalResource("R014", "Federal Service Tribunal Act 1973",
                  "statute", "Establishes Federal Service Tribunal for government employee disputes.",
                  "https://fst.gov.pk/legislation",
                  ["service tribunal", "civil servant", "federal", "dismissal"]),

    LegalResource("R015", "Industrial Relations Act 2012",
                  "statute", "Regulates labor unions, collective bargaining, and industrial disputes.",
                  "https://pakistancode.gov.pk/english/ira-2012",
                  ["labor", "workers", "union", "industrial"]),

    LegalResource("R016", "Anti-Terrorism Act 1997",
                  "statute", "Defines terrorism offences and special procedures for ATC trials.",
                  "https://pakistancode.gov.pk/english/anti-terrorism-act-1997",
                  ["terrorism", "ATA", "ATC", "scheduled offence"]),

    LegalResource("R017", "FIR Registration Form — Police",
                  "form", "First Information Report form for reporting cognizable offences to police.",
                  "https://punjabpolice.gov.pk/forms/fir-form",
                  ["FIR", "criminal", "police", "registration"]),

    LegalResource("R018", "Supreme Court Civil Petition Form",
                  "form", "Official form for filing civil petitions before the Supreme Court.",
                  "https://supremecourt.gov.pk/forms/civil-petition",
                  ["supreme court", "petition", "civil appeal"]),

    LegalResource("R019", "Supreme Court Criminal Petition Form",
                  "form", "Official form for filing criminal leave-to-appeal petitions.",
                  "https://supremecourt.gov.pk/forms/criminal-petition",
                  ["supreme court", "criminal appeal", "leave to appeal"]),

    LegalResource("R020", "Nikahnama — Official Marriage Registration",
                  "form", "Official nikah registration form required by NADRA for Muslim marriages.",
                  "https://nadra.gov.pk/forms/nikahnama",
                  ["marriage", "nikah", "family", "registration"]),
]


# ──────────────────────────────────────────────────
# Main dataset class
# ──────────────────────────────────────────────────

class LegalDataset:
    """
    Legal dataset manager — now powered by 1,414 real Supreme Court judgments.

    Wraps HFDatasetLoader and exposes the same interface as v1
    so all models (TF-IDF, BERT, BiLSTM, RiskPredictor) work unchanged.

    Samsung curriculum: OOP, Pandas DataFrames, NumPy arrays.
    """

    def __init__(self, max_text_length: int = 4000, use_cache: bool = True):
        self.resources = RESOURCES
        self._loader   = HFDatasetLoader(
            max_text_length=max_text_length,
            use_cache=use_cache,
        )
        self._cases_df:     Optional[pd.DataFrame] = None
        self._resources_df: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────
    # DataFrames
    # ──────────────────────────────────────────────

    def get_cases_dataframe(self) -> pd.DataFrame:
        """
        Returns Pandas DataFrame of all 1,414 Supreme Court cases.
        Columns: case_id, title, case_type, jurisdiction, facts,
                 statutes, outcome, risk_level, year, court,
                 citation, url, tags, full_text, hf_embedding
        """
        if self._cases_df is None:
            self._cases_df = self._loader.load()
        return self._cases_df

    def get_resources_dataframe(self) -> pd.DataFrame:
        """Returns DataFrame of 20 legal resources (statutes, forms)."""
        if self._resources_df is None:
            rows = [
                {
                    "resource_id":   r.resource_id,
                    "title":         r.title,
                    "resource_type": r.resource_type,
                    "description":   r.description,
                    "url":           r.url,
                    "tags":          " ".join(r.tags),
                    "full_text":     f"{r.title} {r.description} {' '.join(r.tags)}",
                }
                for r in self.resources
            ]
            self._resources_df = pd.DataFrame(rows)
        return self._resources_df

    # ──────────────────────────────────────────────
    # Training data (NumPy arrays)
    # ──────────────────────────────────────────────

    def get_training_data(self):
        """
        Prepare training data for ML/DL models.

        Returns:
            X        — NumPy array of text strings (1414,)
            y_outcome — Encoded outcome labels (1414,)
            y_risk    — Encoded risk labels (1414,)

        Samsung curriculum: NumPy arrays, label encoding.
        """
        df = self.get_cases_dataframe()

        X = df["full_text"].values   # NumPy array of strings

        outcome_map = {
            "guilty": 0, "acquitted": 1, "settled": 2,
            "pending": 3, "dismissed": 4,
        }
        risk_map = {"high": 2, "medium": 1, "low": 0}

        y_outcome = np.array([outcome_map.get(o, 3) for o in df["outcome"]])
        y_risk    = np.array([risk_map.get(r, 1) for r in df["risk_level"]])

        return X, y_outcome, y_risk

    def get_hf_embeddings(self) -> np.ndarray:
        """
        Returns pre-computed HuggingFace embeddings.
        Shape: (1414, 1024) — L2-normalized.

        These are from mixedbread-ai/mxbai-embed-large-v1.
        Use these directly in BERTEmbedder — no re-encoding needed!

        Samsung curriculum: NumPy matrices, transformer embeddings.
        """
        # Ensure main df is loaded first (embeddings live in it)
        self.get_cases_dataframe()
        return self._loader.get_embeddings_matrix()

    # ──────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────

    def summary(self):
        """
        Print dataset statistics.
        Demonstrates: Pandas value_counts(), string operations.
        """
        self._loader.summary()
        print(f"\n  Legal resources: {len(self.resources)}")


# ──────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    ds = LegalDataset()
    ds.summary()

    df = ds.get_cases_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    X, y_out, y_risk = ds.get_training_data()
    print(f"\nTraining data:")
    print(f"  X shape:        {X.shape}")
    print(f"  y_outcome:      {y_out.shape}  unique={np.unique(y_out)}")
    print(f"  y_risk:         {y_risk.shape}  unique={np.unique(y_risk)}")

    emb = ds.get_hf_embeddings()
    print(f"\nHF Embeddings:  {emb.shape}  dtype={emb.dtype}")
