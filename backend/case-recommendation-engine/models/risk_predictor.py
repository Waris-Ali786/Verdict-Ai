import numpy as np
import pandas as pd
from typing import Dict, Optional
import os, joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

from data.preprocessor import TextPreprocessor
from data.cases_dataset import LegalDataset


class RiskPredictor:
    """
    ML-based risk level and outcome predictor.
    Updated for 1,414 real Supreme Court of Pakistan judgments.

    Models:
      - Random Forest       → risk level (high / medium / low)
      - Gradient Boosting   → likely outcome (guilty / acquitted / settled / pending)

    Features:
      - TF-IDF text features (5,000 terms, sparse matrix)
      - Handcrafted legal features (9 numeric features, dense)
      - Combined via scipy hstack → (1414, 5009)

    Samsung curriculum: scikit-learn, feature engineering, ensemble methods,
                        cross-validation, confusion matrix.
    """

    RISK_LABELS    = ["low", "medium", "high"]
    OUTCOME_LABELS = ["acquitted", "dismissed", "guilty", "pending", "settled"]

    HIGH_RISK_KEYWORDS = [
        "murder", "302", "terrorism", "kidnapping", "narcotics", "anti-terrorism",
        "corruption", "money laundering", "sexual assault", "rape",
        "treason", "gang", "organized crime", "death sentence", "life imprisonment",
        "capital punishment", "scheduled offence",
    ]
    MEDIUM_RISK_KEYWORDS = [
        "fraud", "theft", "robbery", "breach", "contempt", "cybercrime",
        "forgery", "extortion", "dacoity", "hurt",
    ]
    FINANCIAL_KEYWORDS = [
        "fraud", "money", "bank", "financial", "tax", "fbr",
        "income tax", "sales tax", "corruption", "laundering",
    ]

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.dataset      = LegalDataset()

        # Updated: 5,000 features, min_df=2 for real corpus
        self.tfidf = TfidfVectorizer(
            max_features=5_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            strip_accents="unicode",
        )

        # Updated: more estimators, deeper trees
        self.risk_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.outcome_classifier = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )

        self.risk_encoder    = LabelEncoder()
        self.outcome_encoder = LabelEncoder()
        self._is_trained     = False

    # ──────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────

    def _extract_manual_features(self, texts: list) -> np.ndarray:
        """
        Handcrafted legal features from text.
        Same 9 features as v1 — work well on real judgment text.

        Features:
          [0] word count
          [1] statute reference count (section, article, rule, order)
          [2] high-risk keyword count
          [3] medium-risk keyword count
          [4] has murder / homicide indicators (binary)
          [5] has financial crime keywords (binary)
          [6] has family law keywords (binary)
          [7] question marks (disputed facts indicator)
          [8] contains year reference (binary)
        """
        features = []
        for text in texts:
            text_lower = text.lower()
            words      = text_lower.split()
            f = [
                min(len(words), 2000),
                text_lower.count("section") + text_lower.count("article")
                + text_lower.count("rule") + text_lower.count("order"),
                sum(1 for kw in self.HIGH_RISK_KEYWORDS   if kw in text_lower),
                sum(1 for kw in self.MEDIUM_RISK_KEYWORDS if kw in text_lower),
                int(any(k in text_lower for k in ["murder", "302", "killed", "shot", "stabbed"])),
                int(any(k in text_lower for k in self.FINANCIAL_KEYWORDS)),
                int(any(k in text_lower for k in ["divorce", "custody", "marriage", "family"])),
                min(text.count("?"), 10),
                int(any(str(y) in text for y in range(1990, 2026))),
            ]
            features.append(f)
        return np.array(features, dtype=np.float32)

    def _build_feature_matrix(self, texts: list, fit: bool = False):
        """
        Combine TF-IDF (sparse, 5000) + manual (dense, 9) → (n, 5009).
        """
        processed = self.preprocessor.process_batch(texts)

        if fit:
            tfidf_features = self.tfidf.fit_transform(processed)
        else:
            tfidf_features = self.tfidf.transform(processed)

        manual_features = csr_matrix(self._extract_manual_features(texts))
        return hstack([tfidf_features, manual_features])

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────

    def train(self, cv_folds: int = 3):
        """
        Train on 1,414 real Supreme Court judgments.
        No augmentation needed — evaluate with cross-validation.

        cv_folds: number of cross-validation folds for evaluation
        """
        print("[RiskPredictor] Loading dataset...")
        df = self.dataset.get_cases_dataframe()

        texts    = df["full_text"].tolist()
        risks    = df["risk_level"].tolist()
        outcomes = df["outcome"].tolist()

        print(f"[RiskPredictor] Training samples: {len(texts)}")
        print(f"  Risk distribution:    {dict(pd.Series(risks).value_counts())}")
        print(f"  Outcome distribution: {dict(pd.Series(outcomes).value_counts())}")

        # Build features
        print("[RiskPredictor] Building feature matrix (TF-IDF + manual)...")
        X = self._build_feature_matrix(texts, fit=True)
        print(f"[RiskPredictor] Feature matrix shape: {X.shape}")

        # Encode labels
        y_risk    = self.risk_encoder.fit_transform(risks)
        y_outcome = self.outcome_encoder.fit_transform(outcomes)

        # Train models
        print("\n[RiskPredictor] Training Random Forest (risk level)...")
        self.risk_classifier.fit(X, y_risk)

        print("[RiskPredictor] Training Gradient Boosting (outcome)...")
        self.outcome_classifier.fit(X, y_outcome)

        self._is_trained = True

        # Evaluate
        risk_pred    = self.risk_classifier.predict(X)
        outcome_pred = self.outcome_classifier.predict(X)

        risk_acc    = accuracy_score(y_risk, risk_pred)
        outcome_acc = accuracy_score(y_outcome, outcome_pred)

        print(f"\n[RiskPredictor] Training Results:")
        print(f"  Risk accuracy:    {risk_acc:.2%}")
        print(f"  Outcome accuracy: {outcome_acc:.2%}")

        # Cross-validation (more honest evaluation)
        print(f"\n[RiskPredictor] Cross-validation ({cv_folds}-fold)...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        try:
            cv_risk = cross_val_score(self.risk_classifier, X, y_risk, cv=cv, scoring="accuracy")
            print(f"  Risk CV accuracy:    {cv_risk.mean():.2%} ± {cv_risk.std():.2%}")
        except Exception:
            pass

        print(f"\n  Risk Classification Report:")
        print(classification_report(
            y_risk, risk_pred,
            target_names=self.risk_encoder.classes_,
            zero_division=0,
        ))

        return {
            "risk_accuracy":    risk_acc,
            "outcome_accuracy": outcome_acc,
        }

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """Predict risk level and likely outcome for a case."""
        if not self._is_trained:
            return self._rule_based_predict(text)

        X = self._build_feature_matrix([text], fit=False)

        # Risk
        risk_proba = self.risk_classifier.predict_proba(X)[0]
        risk_idx   = int(np.argmax(risk_proba))
        risk_label = self.risk_encoder.classes_[risk_idx]
        risk_conf  = float(risk_proba[risk_idx])

        risk_probs = {
            self.risk_encoder.classes_[i]: round(float(p), 3)
            for i, p in enumerate(risk_proba)
        }

        # Outcome
        out_proba  = self.outcome_classifier.predict_proba(X)[0]
        out_idx    = int(np.argmax(out_proba))
        out_label  = self.outcome_encoder.classes_[out_idx]
        out_conf   = float(out_proba[out_idx])

        out_probs = {
            self.outcome_encoder.classes_[i]: round(float(p), 3)
            for i, p in enumerate(out_proba)
        }

        manual_feats = self._extract_manual_features([text])[0]
        risk_factors = self._explain_risk(manual_feats)

        return {
            "risk_level":             risk_label,
            "risk_confidence":        round(risk_conf, 3),
            "risk_confidence_pct":    f"{risk_conf:.1%}",
            "risk_probabilities":     risk_probs,
            "likely_outcome":         out_label,
            "outcome_confidence":     round(out_conf, 3),
            "outcome_confidence_pct": f"{out_conf:.1%}",
            "outcome_probabilities":  out_probs,
            "risk_factors":           risk_factors,
        }

    def _explain_risk(self, manual_feats: np.ndarray) -> list:
        factors = []
        if manual_feats[2] >= 2:
            factors.append("Multiple high-risk offence keywords detected")
        if manual_feats[4]:
            factors.append("Murder / homicide charges present — severe criminal offence")
        if manual_feats[5]:
            factors.append("Financial crime or corruption elements detected")
        if manual_feats[1] > 5:
            factors.append("Multiple statute references — legally complex case")
        if manual_feats[0] > 500:
            factors.append("Detailed case record suggests complex evidentiary issues")
        if manual_feats[3] >= 2:
            factors.append("Medium-risk offence keywords detected")
        if not factors:
            factors.append("No major aggravating risk factors detected")
        return factors

    def _rule_based_predict(self, text: str) -> Dict:
        """Fallback when model not trained."""
        text_lower = text.lower()
        high = sum(1 for k in self.HIGH_RISK_KEYWORDS if k in text_lower)
        med  = sum(1 for k in self.MEDIUM_RISK_KEYWORDS if k in text_lower)
        if high >= 2:   risk, conf = "high", 0.80
        elif high == 1 or med >= 2: risk, conf = "medium", 0.65
        else:           risk, conf = "low", 0.70
        return {
            "risk_level": risk, "risk_confidence": conf,
            "risk_confidence_pct": f"{conf:.0%}",
            "risk_probabilities": {}, "likely_outcome": "pending",
            "outcome_confidence": 0.4, "outcome_confidence_pct": "40%",
            "outcome_probabilities": {},
            "risk_factors": ["Rule-based fallback — train model for better accuracy"],
        }

    def save(self, path: str = "models/saved/risk_predictor.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "tfidf": self.tfidf,
            "risk_clf": self.risk_classifier,
            "outcome_clf": self.outcome_classifier,
            "risk_encoder": self.risk_encoder,
            "outcome_encoder": self.outcome_encoder,
        }, path)
        print(f"[RiskPredictor] Saved → {path}")

    def load(self, path: str = "models/saved/risk_predictor.joblib"):
        data = joblib.load(path)
        self.tfidf              = data["tfidf"]
        self.risk_classifier    = data["risk_clf"]
        self.outcome_classifier = data["outcome_clf"]
        self.risk_encoder       = data["risk_encoder"]
        self.outcome_encoder    = data["outcome_encoder"]
        self._is_trained        = True
        print(f"[RiskPredictor] Loaded from {path}")
