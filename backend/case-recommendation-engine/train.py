import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from colorama import Fore, Style, init
init(autoreset=True)


def print_section(title):
    print(f"\n{Fore.CYAN}{'═'*58}\n  {title}\n{'═'*58}{Style.RESET_ALL}")


def train_all(use_bert=True, use_bilstm=True, bilstm_epochs=15):
    total_start = time.perf_counter()

    print(f"\n{Fore.GREEN}")
    print("  ██╗     ███████╗ ██████╗ ██╗   ██╗███╗   ███╗")
    print("  ██║     ██╔════╝██╔════╝ ██║   ██║████╗ ████║")
    print("  ██║     █████╗  ██║  ███╗██║   ██║██╔████╔██║")
    print("  ██║     ██╔══╝  ██║   ██║██║   ██║██║╚██╔╝██║")
    print("  ███████╗███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║")
    print(f"{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Legum AI v2 — HuggingFace Dataset Training Pipeline{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Dataset: Ibtehaj10/supreme-court-of-pak-judgments{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Cases: 1,414 real Supreme Court of Pakistan judgments{Style.RESET_ALL}\n")

    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)

    # ── 1. Load HF Dataset ──────────────────────────────────
    print_section("STEP 1 — Loading HuggingFace Dataset")
    from data.hf_dataset_loader import HFDatasetLoader

    t0     = time.perf_counter()
    loader = HFDatasetLoader()
    df     = loader.load()
    loader.summary()
    print(f"\n  {Fore.GREEN}✓ Dataset loaded in {time.perf_counter()-t0:.1f}s{Style.RESET_ALL}")
    print(f"  Cases: {len(df)} | Columns: {list(df.columns)}")

    # ── 2. NLP preprocessing demo ────────────────────────────
    print_section("STEP 2 — NLP Preprocessing Sample")
    from data.preprocessor import TextPreprocessor
    pp = TextPreprocessor()
    sample = df.iloc[0]["full_text"]
    print(f"  Raw (first 100):  {sample[:100].strip()}")
    print(f"  Processed (first 100): {pp.process(sample)[:100]}")
    kw = pp.extract_legal_keywords(sample)
    print(f"  Case types: {kw['case_types']}")
    print(f"  Statutes:   {kw['sections'][:3]}")

    # ── 3. TF-IDF ────────────────────────────────────────────
    print_section("STEP 3 — TF-IDF Retriever (scikit-learn, 30k vocab)")
    from models.tfidf_retriever import TFIDFRetriever
    t0 = time.perf_counter()
    retriever = TFIDFRetriever(top_k=5)
    retriever.fit()
    retriever.save()
    tf_time = time.perf_counter() - t0
    q = "murder section 302 PPC accused Supreme Court"
    cases = retriever.retrieve_similar_cases(q)
    print(f"\n  {Fore.GREEN}✓ TF-IDF fitted in {tf_time:.1f}s{Style.RESET_ALL}")
    print(f"  Vocabulary: {len(retriever.vectorizer.vocabulary_):,} terms")
    print(f"  Test query top result:")
    if cases:
        print(f"    [{cases[0]['similarity_score']}%] {cases[0]['title'][:60]}")
        print(f"    {cases[0]['citation']}")

    # ── 4. Risk Predictor ─────────────────────────────────────
    print_section("STEP 4 — Risk Predictor (Random Forest + Gradient Boosting)")
    from models.risk_predictor import RiskPredictor
    t0 = time.perf_counter()
    predictor = RiskPredictor()
    metrics = predictor.train()
    predictor.save()
    risk_time = time.perf_counter() - t0
    print(f"\n  {Fore.GREEN}✓ Risk Predictor trained in {risk_time:.1f}s{Style.RESET_ALL}")
    print(f"  Risk accuracy:    {metrics['risk_accuracy']:.2%}")
    print(f"  Outcome accuracy: {metrics['outcome_accuracy']:.2%}")

    # ── 5. BiLSTM ─────────────────────────────────────────────
    if use_bilstm:
        print_section("STEP 5 — BiLSTM Classifier (TensorFlow/Keras)")
        from models.bilstm_classifier import BiLSTMClassifier
        t0 = time.perf_counter()
        clf = BiLSTMClassifier()
        history = clf.train(epochs=bilstm_epochs, verbose=1)
        clf.save()
        bilstm_time = time.perf_counter() - t0
        best_acc = max(history.history["accuracy"])
        best_val = max(history.history.get("val_accuracy", [0]))
        print(f"\n  {Fore.GREEN}✓ BiLSTM trained in {bilstm_time:.1f}s{Style.RESET_ALL}")
        print(f"  Best train accuracy: {best_acc:.2%}")
        print(f"  Best val accuracy:   {best_val:.2%}")
        tests = [
            ("murder 302 PPC accused death sentence Supreme Court", "criminal"),
            ("wife divorce family court custody children maintenance", "family"),
            ("civil servant dismissal service tribunal appeal", "service"),
            ("income tax FBR commissioner appeal", "tax"),
        ]
        correct = 0
        print(f"\n  Prediction tests:")
        for text, expected in tests:
            p = clf.predict(text)
            ok = p["predicted_type"] == expected
            correct += int(ok)
            mark = f"{Fore.GREEN}✓" if ok else f"{Fore.RED}✗"
            print(f"  {mark} '{text[:45]}' → {p['predicted_type']} ({p['confidence_pct']}){Style.RESET_ALL}")
        print(f"  Score: {correct}/{len(tests)}")
    else:
        print(f"\n  {Fore.YELLOW}[Skipped] BiLSTM (--no-bilstm){Style.RESET_ALL}")

    # ── 6. BERT ──────────────────────────────────────────────
    if use_bert:
        print_section("STEP 6 — BERT Embeddings (HF Pre-computed, 1024-dim)")
        from models.bert_embedder import BERTEmbedder
        t0 = time.perf_counter()
        bert = BERTEmbedder(top_k=5, use_hf_embeddings=True)
        bert.build_index()
        bert.save_embeddings()
        bert_time = time.perf_counter() - t0
        info = bert.get_embedding_info("murder case Section 302 Karachi")
        print(f"\n  {Fore.GREEN}✓ BERT index built in {bert_time:.1f}s{Style.RESET_ALL}")
        print(f"  Embedding dim:  {info['dimensions']}")
        print(f"  Vector norm:    {info['norm']} (1.0 = normalized)")
        bert_cases = bert.retrieve_similar_cases("murder section 302 PPC accused")
        if bert_cases:
            print(f"  Top BERT result: [{bert_cases[0]['similarity_score']}%] {bert_cases[0]['title'][:55]}")
    else:
        print(f"\n  {Fore.YELLOW}[Skipped] BERT (--no-bert){Style.RESET_ALL}")

    # ── Summary ──────────────────────────────────────────────
    total_time = time.perf_counter() - total_start
    print_section("TRAINING COMPLETE")
    print(f"\n  {Fore.GREEN}All models trained and saved!{Style.RESET_ALL}")
    print(f"  Dataset: 1,414 real Supreme Court judgments")
    print(f"  Total time: {total_time:.0f}s\n")

    saved = []
    for root, _, files in os.walk("models/saved"):
        for f in files:
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            saved.append((p, sz))

    if saved:
        print("  Saved model files:")
        for path, sz in saved:
            label = f"{sz/1024:.0f} KB" if sz < 1024*1024 else f"{sz/1024/1024:.1f} MB"
            print(f"    • {path}  ({label})")

    print(f"""
  {Fore.CYAN}Next steps:
    python main.py --demo           # CLI demo
    python api/app.py               # Web UI at http://localhost:5000
    python tests/test_engine.py     # Unit tests{Style.RESET_ALL}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legum AI v2 — Train all models")
    parser.add_argument("--no-bert",       action="store_true")
    parser.add_argument("--no-bilstm",     action="store_true")
    parser.add_argument("--bilstm-epochs", type=int, default=15)
    args = parser.parse_args()
    train_all(
        use_bert=not args.no_bert,
        use_bilstm=not args.no_bilstm,
        bilstm_epochs=args.bilstm_epochs,
    )
