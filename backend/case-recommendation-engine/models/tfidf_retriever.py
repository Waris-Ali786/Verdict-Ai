import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import joblib
import os

from data.preprocessor import TextPreprocessor
from data.cases_dataset import LegalDataset


class TFIDFRetriever:
    """
    TF-IDF based legal case and resource retriever.
    Scaled up for 1,414 real Supreme Court judgments.

    Samsung curriculum: scikit-learn, TF-IDF, cosine similarity, OOP.
    """

    def __init__(
        self,
        max_features: int = 30000,   # v1: 5000 → v2: 30000
        ngram_range: Tuple[int, int] = (1, 2),
        top_k: int = 5,
        min_df: int = 2,             # v1: 1 → v2: 2 (ignore very rare terms)
    ):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.top_k        = top_k
        self.min_df       = min_df

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # log(1+tf) — handles long judgments
            min_df=min_df,
            max_df=0.95,             # ignore terms in >95% of docs (too common)
            analyzer="word",
            strip_accents="unicode",
        )

        self.preprocessor = TextPreprocessor()
        self.dataset      = LegalDataset()

        self._case_matrix     = None
        self._resource_matrix = None
        self._case_texts: List[str]     = []
        self._resource_texts: List[str] = []
        self._is_fitted = False

    # ──────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────

    def fit(self):
        """
        Build TF-IDF matrices for 1,414 real cases + 20 resources.

        With real data the vocabulary is ~80k unique terms — we cap at
        max_features=30,000 to keep it manageable while preserving
        the most informative legal terms.
        """
        print("[TF-IDF] Loading Supreme Court dataset...")
        cases_df     = self.dataset.get_cases_dataframe()
        resources_df = self.dataset.get_resources_dataframe()

        print(f"[TF-IDF] Preprocessing {len(cases_df)} case texts...")
        self._case_texts     = cases_df["full_text"].tolist()
        self._resource_texts = resources_df["full_text"].tolist()

        processed_cases     = self.preprocessor.process_batch(self._case_texts)
        processed_resources = self.preprocessor.process_batch(self._resource_texts)

        # Fit on combined corpus
        all_texts = processed_cases + processed_resources
        print(f"[TF-IDF] Fitting on {len(all_texts)} documents...")
        self.vectorizer.fit(all_texts)

        # Transform
        print("[TF-IDF] Transforming case corpus...")
        self._case_matrix     = self.vectorizer.transform(processed_cases)
        self._resource_matrix = self.vectorizer.transform(processed_resources)

        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"[TF-IDF] Done.")
        print(f"  Vocabulary:       {vocab_size:,} terms")
        print(f"  Case matrix:      {self._case_matrix.shape}")
        print(f"  Resource matrix:  {self._resource_matrix.shape}")

    # ──────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────

    def retrieve_similar_cases(self, query: str) -> List[Dict]:
        """
        Retrieve most similar Supreme Court cases using cosine similarity.

        With 1,414 real cases the results will be actual judgments
        from the Supreme Court of Pakistan.
        """
        if not self._is_fitted:
            self.fit()

        processed_query = self.preprocessor.process(query)
        query_vector    = self.vectorizer.transform([processed_query])
        similarities    = cosine_similarity(query_vector, self._case_matrix)[0]
        top_indices     = np.argsort(similarities)[::-1][: self.top_k]

        cases_df = self.dataset.get_cases_dataframe()
        results  = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.01:
                continue
            row = cases_df.iloc[idx]
            results.append({
                "case_id":        row["case_id"],
                "title":          row["title"],
                "case_type":      row["case_type"],
                "jurisdiction":   row["jurisdiction"],
                "court":          row["court"],
                "citation":       row["citation"],
                "outcome":        row["outcome"],
                "risk_level":     row["risk_level"],
                "year":           int(row["year"]),
                "url":            row["url"],
                "facts_snippet":  row["facts"][:250] + "...",
                "statutes":       [s.strip() for s in row["statutes"].split("|") if s.strip()],
                "similarity_score": round(score * 100, 1),
                "source":         "TF-IDF Retriever",
            })

        return results

    def retrieve_similar_resources(self, query: str) -> List[Dict]:
        """Retrieve relevant legal resources for the query."""
        if not self._is_fitted:
            self.fit()

        processed_query  = self.preprocessor.process(query)
        query_vector     = self.vectorizer.transform([processed_query])
        similarities     = cosine_similarity(query_vector, self._resource_matrix)[0]
        top_indices      = np.argsort(similarities)[::-1][: self.top_k]

        resources_df = self.dataset.get_resources_dataframe()
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.01:
                continue
            row = resources_df.iloc[idx]
            results.append({
                "resource_id":   row["resource_id"],
                "title":         row["title"],
                "resource_type": row["resource_type"],
                "description":   row["description"],
                "url":           row["url"],
                "tags":          row["tags"].split(),
                "similarity_score": round(score * 100, 1),
                "source":        "TF-IDF Retriever",
            })

        return results

    def get_top_terms(self, text: str, n: int = 10) -> List[Tuple[str, float]]:
        """Educational: show top TF-IDF weighted terms for a text."""
        if not self._is_fitted:
            self.fit()
        processed      = self.preprocessor.process(text)
        vector         = self.vectorizer.transform([processed])
        feature_names  = self.vectorizer.get_feature_names_out()
        scores         = vector.toarray()[0]
        top_indices    = np.argsort(scores)[::-1][:n]
        return [(feature_names[i], round(float(scores[i]), 4))
                for i in top_indices if scores[i] > 0]

    def save(self, path: str = "models/saved/tfidf_retriever.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "vectorizer":      self.vectorizer,
            "case_matrix":     self._case_matrix,
            "resource_matrix": self._resource_matrix,
            "case_texts":      self._case_texts,
            "resource_texts":  self._resource_texts,
        }, path)
        print(f"[TF-IDF] Saved → {path}")

    def load(self, path: str = "models/saved/tfidf_retriever.joblib"):
        data = joblib.load(path)
        self.vectorizer       = data["vectorizer"]
        self._case_matrix     = data["case_matrix"]
        self._resource_matrix = data["resource_matrix"]
        self._case_texts      = data["case_texts"]
        self._resource_texts  = data["resource_texts"]
        self._is_fitted       = True
        print(f"[TF-IDF] Loaded from {path}")
