import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np


class TestHFDatasetLoader(unittest.TestCase):
    """Tests for HuggingFace dataset loading."""

    @classmethod
    def setUpClass(cls):
        from data.hf_dataset_loader import HFDatasetLoader
        cls.loader = HFDatasetLoader(max_text_length=1000)
        cls.df = cls.loader.load()

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.df), 100)

    def test_dataset_has_required_columns(self):
        for col in ["case_id", "title", "case_type", "outcome", "risk_level",
                    "year", "full_text", "citation", "url"]:
            self.assertIn(col, self.df.columns)

    def test_case_types_valid(self):
        valid = {"criminal", "family", "civil", "corporate",
                 "constitutional", "service", "tax"}
        for ct in self.df["case_type"]:
            self.assertIn(ct, valid)

    def test_risk_levels_valid(self):
        for r in self.df["risk_level"]:
            self.assertIn(r, {"high", "medium", "low"})

    def test_outcomes_valid(self):
        for o in self.df["outcome"]:
            self.assertIn(o, {"guilty", "acquitted", "settled", "pending", "dismissed"})

    def test_embeddings_shape(self):
        emb = self.loader.get_embeddings_matrix()
        self.assertEqual(len(emb), len(self.df))
        self.assertEqual(emb.shape[1], 1024)
        self.assertEqual(emb.dtype, np.float32)

    def test_embeddings_normalized(self):
        emb = self.loader.get_embeddings_matrix()
        norms = np.linalg.norm(emb[:10], axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=3)

    def test_year_range(self):
        for y in self.df["year"]:
            self.assertGreaterEqual(y, 1947)
            self.assertLessEqual(y, 2025)


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        from data.preprocessor import TextPreprocessor
        self.pp = TextPreprocessor()

    def test_clean_lowercases(self):
        self.assertEqual(self.pp.clean("MURDER Section 302"),
                         self.pp.clean("MURDER Section 302").lower())

    def test_process_returns_string(self):
        result = self.pp.process("SUPREME COURT OF PAKISTAN accused Section 302 PPC murder")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_extract_case_type_criminal(self):
        kw = self.pp.extract_legal_keywords("accused charged murder FIR section 302 PPC prosecution")
        self.assertIn("criminal", kw["case_types"])

    def test_extract_case_type_service(self):
        kw = self.pp.extract_legal_keywords("civil servant dismissal from service tribunal government employee")
        self.assertIn("service", kw["case_types"])

    def test_extract_jurisdiction_federal(self):
        kw = self.pp.extract_legal_keywords("Supreme Court of Pakistan Islamabad federal")
        self.assertIn("federal", kw["jurisdictions"])


class TestTFIDFRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.tfidf_retriever import TFIDFRetriever
        cls.retriever = TFIDFRetriever(top_k=3)
        cls.retriever.fit()

    def test_fit_creates_matrices(self):
        self.assertIsNotNone(self.retriever._case_matrix)
        self.assertIsNotNone(self.retriever._resource_matrix)

    def test_case_matrix_shape(self):
        # Should have ~1414 rows
        self.assertGreater(self.retriever._case_matrix.shape[0], 100)

    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve_similar_cases("murder section 302 PPC")
        self.assertIsInstance(results, list)

    def test_scores_in_range(self):
        results = self.retriever.retrieve_similar_cases("murder accused criminal")
        for r in results:
            self.assertGreaterEqual(r["similarity_score"], 0)
            self.assertLessEqual(r["similarity_score"], 100)

    def test_results_have_required_fields(self):
        results = self.retriever.retrieve_similar_cases("divorce family court custody")
        if results:
            for field in ["case_id", "title", "citation", "url", "similarity_score"]:
                self.assertIn(field, results[0])

    def test_resource_retrieval(self):
        results = self.retriever.retrieve_similar_resources("murder criminal law")
        self.assertIsInstance(results, list)
        if results:
            self.assertIn("resource_type", results[0])


class TestRiskPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.risk_predictor import RiskPredictor
        cls.pred = RiskPredictor()
        cls.pred.train()

    def test_predict_returns_dict(self):
        r = self.pred.predict("murder accused section 302 PPC")
        self.assertIsInstance(r, dict)

    def test_predict_has_required_keys(self):
        r = self.pred.predict("divorce custody family court")
        for key in ["risk_level", "risk_confidence", "likely_outcome",
                    "outcome_confidence", "risk_factors"]:
            self.assertIn(key, r)

    def test_risk_level_valid(self):
        r = self.pred.predict("income tax FBR appeal")
        self.assertIn(r["risk_level"], ["high", "medium", "low"])

    def test_confidence_range(self):
        r = self.pred.predict("civil servant dismissal service tribunal")
        self.assertGreaterEqual(r["risk_confidence"], 0.0)
        self.assertLessEqual(r["risk_confidence"], 1.0)

    def test_high_risk_keywords(self):
        r = self.pred.predict("murder terrorism kidnapping death sentence narcotics trafficking")
        self.assertIn(r["risk_level"], ["high", "medium"])

    def test_risk_factors_not_empty(self):
        r = self.pred.predict("any legal text")
        self.assertIsInstance(r["risk_factors"], list)
        self.assertGreater(len(r["risk_factors"]), 0)


class TestBiLSTMClassifier(unittest.TestCase):
    def setUp(self):
        from models.bilstm_classifier import BiLSTMClassifier
        self.clf = BiLSTMClassifier()

    def test_rule_based_criminal(self):
        r = self.clf._rule_based_fallback("murder FIR accused section 302 PPC criminal")
        self.assertEqual(r["predicted_type"], "criminal")

    def test_rule_based_family(self):
        r = self.clf._rule_based_fallback("divorce khul marriage nikah family court")
        self.assertEqual(r["predicted_type"], "family")

    def test_rule_based_service(self):
        r = self.clf._rule_based_fallback("civil servant dismissal from service tribunal government employee")
        self.assertEqual(r["predicted_type"], "service")

    def test_rule_based_tax(self):
        r = self.clf._rule_based_fallback("income tax FBR commissioner appeal assessment")
        self.assertEqual(r["predicted_type"], "tax")

    def test_confidence_range(self):
        r = self.clf._rule_based_fallback("some legal text")
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)

    def test_all_probs_sum(self):
        r = self.clf._rule_based_fallback("murder accused criminal section 302")
        total = sum(r["all_probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=1)


if __name__ == "__main__":
    print("=" * 58)
    print("  Legum AI v2 — Unit Tests (HuggingFace Dataset)")
    print("=" * 58)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestHFDatasetLoader,
        TestTextPreprocessor,
        TestTFIDFRetriever,
        TestRiskPredictor,
        TestBiLSTMClassifier,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
