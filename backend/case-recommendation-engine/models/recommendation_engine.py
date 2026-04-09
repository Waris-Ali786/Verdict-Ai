import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from data.preprocessor import TextPreprocessor
from models.tfidf_retriever import TFIDFRetriever
from models.bert_embedder import BERTEmbedder
from models.bilstm_classifier import BiLSTMClassifier
from models.risk_predictor import RiskPredictor


# ──────────────────────────────────────────────────
# OOP: Result data structures
# ──────────────────────────────────────────────────

@dataclass
class RecommendedCase:
    """A similar case recommended by the engine."""
    case_id: str
    title: str
    case_type: str
    jurisdiction: str
    court: str
    citation: str
    outcome: str
    risk_level: str
    year: int
    url: str
    facts_snippet: str
    statutes: List[str]
    relevance_score: float          # 0–100
    retrieval_method: str           # "TF-IDF" | "BERT" | "Combined"
    legal_factors_matched: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class RecommendedResource:
    """A legal resource (statute, form, guideline) recommended."""
    resource_id: str
    title: str
    resource_type: str
    description: str
    url: str
    tags: List[str]
    relevance_score: float


@dataclass
class CaseAnalysisResult:
    """Complete analysis output from the recommendation engine."""
    # Input
    input_text_snippet: str
    input_length: int

    # Classification
    detected_case_type: str
    case_type_confidence: float
    case_type_probabilities: Dict[str, float]

    # Risk prediction
    risk_level: str
    risk_confidence: float
    likely_outcome: str
    outcome_confidence: float
    risk_factors: List[str]
    outcome_probabilities: Dict[str, float]

    # Recommendations
    similar_cases: List[RecommendedCase]
    recommended_resources: List[RecommendedResource]

    # NLP insights
    key_legal_issues: List[str]
    detected_statutes: List[str]
    detected_jurisdiction: str

    # Metadata
    processing_time_sec: float
    models_used: List[str]


# ──────────────────────────────────────────────────
# Master Engine
# ──────────────────────────────────────────────────

class RecommendationEngine:
    """
    Legum AI — Master Recommendation Engine

    Orchestrates all AI components:
      - TextPreprocessor    (NLP)
      - TFIDFRetriever      (Machine Learning — sklearn)
      - BERTEmbedder        (Transformers — Sentence-BERT)
      - BiLSTMClassifier    (Deep Learning — TensorFlow)
      - RiskPredictor       (ML — Random Forest + Gradient Boosting)

    Samsung curriculum: OOP design, system integration, data fusion
    """

    def __init__(self, use_bert: bool = True, use_bilstm: bool = True):
        """
        Args:
            use_bert:   Include Sentence-BERT retrieval (slower but smarter)
            use_bilstm: Use BiLSTM classifier (requires training)
        """
        self.use_bert   = use_bert
        self.use_bilstm = use_bilstm

        print("=" * 55)
        print(" Legum AI — Recommendation Engine Initializing")
        print("=" * 55)

        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.tfidf         = TFIDFRetriever(top_k=5)
        self.risk_predictor = RiskPredictor()
        self.classifier     = BiLSTMClassifier()

        if self.use_bert:
            self.bert = BERTEmbedder(top_k=5)

        self._models_initialized = False

    def initialize(self):
        """
        Load and fit all models.
        Call this once at startup before handling requests.
        """
        print("\n[Engine] Initializing all AI models...")
        start = time.time()

        # 1. Fit TF-IDF on dataset
        self.tfidf.fit()

        # 2. Build BERT embedding index
        if self.use_bert:
            self.bert.build_index()

        # 3. Train risk predictor
        self.risk_predictor.train()

        # 4. Train BiLSTM classifier
        if self.use_bilstm:
            self.classifier.train(epochs=10, verbose=0)

        elapsed = time.time() - start
        self._models_initialized = True
        print(f"\n[Engine] All models ready in {elapsed:.1f}s")
        print("=" * 55)

    # ──────────────────────────────────────────────────
    # Main analysis method
    # ──────────────────────────────────────────────────
    def analyze(self, text: str) -> CaseAnalysisResult:
        """
        Full pipeline: text → complete analysis + recommendations.

        Steps:
          1.  Validate & clean input
          2.  Extract NLP keywords
          3.  Classify case type (BiLSTM or rule-based)
          4.  Predict risk & outcome (Random Forest / GB)
          5.  Retrieve similar cases (TF-IDF)
          6.  Retrieve similar cases (BERT)
          7.  Retrieve relevant resources (TF-IDF + BERT)
          8.  Fuse & deduplicate results
          9.  Build and return CaseAnalysisResult
        """
        if not self._models_initialized:
            self.initialize()

        start_time = time.time()
        models_used = ["TextPreprocessor", "TF-IDF Retriever", "Risk Predictor (RF+GB)"]

        # ── Step 1: Input validation ──
        text = text.strip()
        if len(text) < 20:
            raise ValueError("Case text too short. Please provide at least 20 characters.")

        # ── Step 2: NLP keyword extraction ──
        keywords = self.preprocessor.extract_legal_keywords(text)

        # ── Step 3: Case type classification ──
        if self.use_bilstm:
            classification = self.classifier.predict(text)
            models_used.append("BiLSTM Classifier")
        else:
            classification = self.classifier._rule_based_fallback(text)
            models_used.append("Rule-Based Classifier")

        # ── Step 4: Risk & outcome prediction ──
        risk_result = self.risk_predictor.predict(text)

        # ── Step 5: TF-IDF case retrieval ──
        tfidf_cases     = self.tfidf.retrieve_similar_cases(text)
        tfidf_resources = self.tfidf.retrieve_similar_resources(text)

        # ── Step 6: BERT semantic retrieval ──
        bert_cases, bert_resources = [], []
        if self.use_bert:
            bert_cases     = self.bert.retrieve_similar_cases(text)
            bert_resources = self.bert.retrieve_similar_resources(text)
            models_used.append("Sentence-BERT Embeddings")

        # ── Step 7: Fuse case results ──
        fused_cases     = self._fuse_cases(
            tfidf_cases, 
            bert_cases, 
            detected_type=classification["predicted_type"],
            detected_statutes=keywords.get("sections", []),
            detected_jurisdiction=jurisdiction
        )
        fused_resources = self._fuse_resources(tfidf_resources, bert_resources)

        # ── Step 8: Build result ──
        processing_time = round(time.time() - start_time, 2)

        similar_cases = [
            RecommendedCase(
                case_id          = c["case_id"],
                title            = c["title"],
                case_type        = c["case_type"],
                jurisdiction     = c["jurisdiction"],
                court            = c["court"],
                citation         = c["citation"],
                outcome          = c["outcome"],
                risk_level       = c["risk_level"],
                year             = c["year"],
                url              = c["url"],
                facts_snippet    = c.get("facts_snippet", ""),
                statutes         = c.get("statutes", []),
                relevance_score  = c["similarity_score"],
                retrieval_method = c.get("source", "Combined"),
                legal_factors_matched = c.get("legal_factors", []),
                tags             = keywords.get("keywords", []),
            )
            for c in fused_cases
        ]

        recommended_resources = [
            RecommendedResource(
                resource_id    = r["resource_id"],
                title          = r["title"],
                resource_type  = r["resource_type"],
                description    = r["description"],
                url            = r["url"],
                tags           = r.get("tags", []),
                relevance_score = r["similarity_score"],
            )
            for r in fused_resources
        ]

        # Detect primary jurisdiction
        jurs = keywords.get("jurisdictions", [])
        jurisdiction = jurs[0] if jurs else "federal"

        return CaseAnalysisResult(
            # Input
            input_text_snippet    = text[:300] + ("..." if len(text) > 300 else ""),
            input_length          = len(text),
            # Classification
            detected_case_type    = classification["predicted_type"],
            case_type_confidence  = classification["confidence"],
            case_type_probabilities = classification["all_probabilities"],
            # Risk
            risk_level            = risk_result["risk_level"],
            risk_confidence       = risk_result["risk_confidence"],
            likely_outcome        = risk_result["likely_outcome"],
            outcome_confidence    = risk_result["outcome_confidence"],
            risk_factors          = risk_result["risk_factors"],
            outcome_probabilities = risk_result.get("outcome_probabilities", {}),
            # Recommendations
            similar_cases         = similar_cases,
            recommended_resources = recommended_resources,
            # NLP
            key_legal_issues      = keywords.get("keywords", [])[:6],
            detected_statutes     = keywords.get("sections", []),
            detected_jurisdiction = jurisdiction,
            # Meta
            processing_time_sec   = processing_time,
            models_used           = models_used,
        )

    # ──────────────────────────────────────────────────
    # Result fusion
    # ──────────────────────────────────────────────────
    def _fuse_cases(
        self, 
        tfidf_cases: list, 
        bert_cases: list,
        detected_type: str = "",
        detected_statutes: List[str] = None,
        detected_jurisdiction: str = ""
    ) -> list:
        """
        Merge TF-IDF and BERT results, deduplicate, and re-rank using legal factors.

        Legal Scoring Rubric:
          - Base Similarity: 40% TF-IDF + 60% BERT
          - Statute Match: +15 points (if any statute matches)
          - Case Type Match: +10 points
          - Jurisdiction Match: +5 points
          - Recency Boost: +5 points (if year >= 2018)

        Samsung curriculum: Data fusion, heuristic scoring, list processing.
        """
        scores: Dict[str, dict] = {}
        detected_statutes = set(detected_statutes or [])

        for c in tfidf_cases:
            cid = c["case_id"]
            scores[cid] = {**c, "tfidf_score": c["similarity_score"], "bert_score": 0}

        for c in bert_cases:
            cid = c["case_id"]
            if cid in scores:
                scores[cid]["bert_score"] = c["similarity_score"]
                combined = 0.4 * scores[cid]["tfidf_score"] + 0.6 * c["similarity_score"]
                scores[cid]["similarity_score"] = round(combined, 1)
                scores[cid]["source"] = "TF-IDF + BERT Combined"
            else:
                scores[cid] = {**c, "tfidf_score": 0, "bert_score": c["similarity_score"]}

        # Apply Legal Factor Boosts
        for cid, c in scores.items():
            legal_factors = []
            boost = 0
            
            # 1. Statute Match
            case_statutes = set(c.get("statutes", []))
            matches = detected_statutes.intersection(case_statutes)
            if matches:
                boost += 15
                legal_factors.append(f"Statute Match: {', '.join(list(matches)[:2])}")
            
            # 2. Case Type Match
            if c["case_type"].lower() == detected_type.lower():
                boost += 10
                legal_factors.append(f"Case Type: {c['case_type']}")
            
            # 3. Jurisdiction Match
            if c["jurisdiction"].lower() == detected_jurisdiction.lower():
                boost += 5
                legal_factors.append(f"Jurisdiction: {c['jurisdiction']}")
            
            # 4. Recency Boost
            if c["year"] >= 2018:
                boost += 5
                legal_factors.append(f"Recent Precedent ({c['year']})")
            
            c["similarity_score"] = min(100.0, c["similarity_score"] + boost)
            c["legal_factors"] = legal_factors

        # Sort by final score
        fused = sorted(scores.values(), key=lambda x: x["similarity_score"], reverse=True)
        return fused[:5]

    def _fuse_resources(self, tfidf_res: list, bert_res: list) -> list:
        """Same fusion logic for resources."""
        scores: Dict[str, dict] = {}

        for r in tfidf_res:
            rid = r["resource_id"]
            scores[rid] = {**r, "tfidf_score": r["similarity_score"]}

        for r in bert_res:
            rid = r["resource_id"]
            if rid in scores:
                combined = 0.4 * scores[rid]["similarity_score"] + 0.6 * r["similarity_score"]
                scores[rid]["similarity_score"] = round(combined, 1)
            else:
                scores[rid] = {**r}

        return sorted(scores.values(), key=lambda x: x["similarity_score"], reverse=True)[:6]

    # ──────────────────────────────────────────────────
    # Output formatting
    # ──────────────────────────────────────────────────
    def to_dict(self, result: CaseAnalysisResult) -> dict:
        """Convert CaseAnalysisResult dataclass to JSON-serializable dict."""
        return {
            "input_snippet":        result.input_text_snippet,
            "input_length":         result.input_length,
            "detected_case_type":   result.detected_case_type,
            "case_type_confidence": result.case_type_confidence,
            "case_type_probabilities": result.case_type_probabilities,
            "risk_level":           result.risk_level,
            "risk_confidence":      result.risk_confidence,
            "likely_outcome":       result.likely_outcome,
            "outcome_confidence":   result.outcome_confidence,
            "risk_factors":         result.risk_factors,
            "outcome_probabilities": result.outcome_probabilities,
            "key_legal_issues":     result.key_legal_issues,
            "detected_statutes":    result.detected_statutes,
            "detected_jurisdiction": result.detected_jurisdiction,
            "similar_cases": [
                {
                    "case_id":        c.case_id,
                    "title":          c.title,
                    "case_type":      c.case_type,
                    "jurisdiction":   c.jurisdiction,
                    "court":          c.court,
                    "citation":       c.citation,
                    "outcome":        c.outcome,
                    "risk_level":     c.risk_level,
                    "year":           c.year,
                    "url":            c.url,
                    "facts_snippet":  c.facts_snippet,
                    "statutes":       c.statutes,
                    "relevance_score": c.relevance_score,
                    "retrieval_method": c.retrieval_method,
                    "legal_factors_matched": c.legal_factors_matched,
                }
                for c in result.similar_cases
            ],
            "recommended_resources": [
                {
                    "resource_id":   r.resource_id,
                    "title":         r.title,
                    "resource_type": r.resource_type,
                    "description":   r.description,
                    "url":           r.url,
                    "tags":          r.tags,
                    "relevance_score": r.relevance_score,
                }
                for r in result.recommended_resources
            ],
            "processing_time_sec": result.processing_time_sec,
            "models_used":         result.models_used,
        }
