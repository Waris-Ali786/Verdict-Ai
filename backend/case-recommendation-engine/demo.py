import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from colorama import Fore, Back, Style, init
init(autoreset=True)


# ──────────────────────────────────────────────────
# Terminal formatting helpers
# ──────────────────────────────────────────────────

def banner():
    print(f"\n{Fore.GREEN}")
    print("  ██╗     ███████╗ ██████╗ ██╗   ██╗███╗   ███╗  ██╗   ██╗██████╗")
    print("  ██║     ██╔════╝██╔════╝ ██║   ██║████╗ ████║  ██║   ██║╚════██╗")
    print("  ██║     █████╗  ██║  ███╗██║   ██║██╔████╔██║  ██║   ██║ █████╔╝")
    print("  ██║     ██╔══╝  ██║   ██║██║   ██║██║╚██╔╝██║  ╚██╗ ██╔╝██╔═══╝")
    print("  ███████╗███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║   ╚████╔╝ ███████╗")
    print("  ╚══════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝    ╚═══╝  ╚══════╝")
    print(f"{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Legum AI v2 — Real Dataset Edition{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Samsung AI Training Program — Final Project{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Dataset: 1,414 Supreme Court of Pakistan Judgments (HuggingFace){Style.RESET_ALL}")
    print(f"\n  {'─'*60}")


def section(title):
    print(f"\n{Fore.YELLOW}  ▸ {title}{Style.RESET_ALL}")
    print(f"  {'─'*55}")


def kv(key, value, color=Fore.WHITE):
    print(f"  {Fore.WHITE}{key:<24}{Style.RESET_ALL}{color}{value}{Style.RESET_ALL}")


def bullet(text, color=Fore.CYAN):
    print(f"  {color}•{Style.RESET_ALL} {text}")


def score_bar(score, width=20):
    filled = int(score / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = Fore.GREEN if score >= 70 else (Fore.YELLOW if score >= 40 else Fore.RED)
    return f"{color}{bar}{Style.RESET_ALL} {score:.1f}%"


def risk_color(level):
    return {
        "high": Fore.RED, "medium": Fore.YELLOW, "low": Fore.GREEN
    }.get(level, Fore.WHITE)


DEMO_CASES = [
    {
        "label": "Criminal — Murder Appeal (Section 302 PPC)",
        "text": (
            "The accused Khalid Rehman has filed an appeal before the Supreme Court of Pakistan "
            "against his conviction under Section 302 PPC for the murder of the deceased Shahid "
            "Mehmood in Karachi. The Sindh High Court upheld the conviction and death sentence. "
            "Defense counsel argues misidentification and absence of motive. Prosecution relies "
            "on eyewitness testimony and ballistics report from the crime scene."
        ),
    },
    {
        "label": "Service — Civil Servant Wrongful Dismissal",
        "text": (
            "A civil servant employed with Pakistan Railways was dismissed from service vide "
            "order without issuance of show-cause notice and without affording opportunity of "
            "hearing, violating principles of natural justice. The Federal Service Tribunal "
            "dismissed the service appeal. Present appeal before Supreme Court seeks reinstatement "
            "with back pay, seniority restoration, and recovery of pensionary benefits."
        ),
    },
    {
        "label": "Family — Divorce, Custody and Maintenance",
        "text": (
            "Wife Ayesha Bibi filed for dissolution of marriage by Khul in Family Court Lahore "
            "under Muslim Family Laws Ordinance 1961 citing persistent cruelty and domestic violence. "
            "Husband contesting the divorce and seeking exclusive custody of two minor children "
            "aged 4 and 7. Case involves questions of dower recovery, monthly maintenance under "
            "West Pakistan Family Courts Act 1964, and guardianship under Guardian and Wards Act 1890."
        ),
    },
    {
        "label": "Tax — Income Tax FBR Appeal",
        "text": (
            "The taxpayer has challenged the assessment order passed by the Commissioner Inland "
            "Revenue under the Income Tax Ordinance 2001, adding unexplained income of PKR 50 million "
            "to taxable income without proper basis. The FBR issued a show-cause notice. The appellate "
            "tribunal dismissed the appeal. Present appeal before Supreme Court raises questions of "
            "burden of proof and principles of natural justice in tax proceedings."
        ),
    },
]


def run_demo(use_bert=True, custom_query=None):
    banner()

    # ── Step 1: Load real dataset ──────────────────────────────
    section("STEP 1 — HuggingFace Dataset (1,414 Real SC Judgments)")
    from data.hf_dataset_loader import HFDatasetLoader
    import numpy as np

    t0     = time.perf_counter()
    loader = HFDatasetLoader()
    df     = loader.load()
    t_load = time.perf_counter() - t0

    kv("Source:",    "Ibtehaj10/supreme-court-of-pak-judgments", Fore.CYAN)
    kv("Cases:",     f"{len(df):,} real Supreme Court judgments", Fore.GREEN)
    kv("Loaded in:", f"{t_load:.1f}s", Fore.CYAN)
    kv("Year range:", f"{df['year'].min()} – {df['year'].max()}", Fore.CYAN)

    print(f"\n  Case type distribution (Pandas value_counts):")
    for ct, count in df["case_type"].value_counts().items():
        bar = "█" * (count // 8)
        print(f"  {Fore.WHITE}{ct:<16}{Style.RESET_ALL} {Fore.TEAL}{bar}{Style.RESET_ALL} {count}")

    # Pre-computed embeddings info
    emb = loader.get_embeddings_matrix()
    print(f"\n  Pre-computed embeddings: {emb.shape}  (mxbai-embed-large-v1)")
    print(f"  Sample vector norm:      {np.linalg.norm(emb[0]):.4f} (L2 normalized)")

    # ── Step 2: NLP preprocessing ──────────────────────────────
    section("STEP 2 — NLP Preprocessing on Real Judgment Text")
    from data.preprocessor import TextPreprocessor

    pp   = TextPreprocessor()
    real = df.iloc[0]["full_text"]

    print(f"  {Fore.WHITE}Raw (first 120 chars):{Style.RESET_ALL}")
    print(f"  {real[:120].strip()}")
    print(f"\n  {Fore.WHITE}After NLP pipeline:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{pp.process(real)[:120]}{Style.RESET_ALL}")

    kw = pp.extract_legal_keywords(real)
    print(f"\n  Detected case types:  {Fore.CYAN}{kw['case_types']}{Style.RESET_ALL}")
    print(f"  Detected statutes:    {Fore.CYAN}{kw['sections'][:4]}{Style.RESET_ALL}")
    print(f"  Top keywords:         {Fore.CYAN}{kw['keywords'][:6]}{Style.RESET_ALL}")

    # ── Step 3: TF-IDF ─────────────────────────────────────────
    section("STEP 3 — TF-IDF Retrieval (scikit-learn, 30,000 term vocab)")
    from models.tfidf_retriever import TFIDFRetriever

    t0        = time.perf_counter()
    retriever = TFIDFRetriever(top_k=3)
    retriever.fit()
    t_tfidf   = time.perf_counter() - t0

    kv("Fitted in:",  f"{t_tfidf:.1f}s", Fore.GREEN)
    kv("Vocabulary:", f"{len(retriever.vectorizer.vocabulary_):,} terms", Fore.CYAN)

    query = custom_query or DEMO_CASES[0]["text"]
    print(f"\n  Query: '{query[:80]}...'")

    print(f"\n  Top TF-IDF weighted terms:")
    for term, score in retriever.get_top_terms(query, n=8):
        bar = "█" * int(score * 600)
        print(f"  {Fore.WHITE}{term:<20}{Style.RESET_ALL} {Fore.TEAL}{bar}{Style.RESET_ALL} {score:.4f}")

    print(f"\n  {Fore.YELLOW}Similar cases from 1,414 real SC judgments:{Style.RESET_ALL}")
    for c in retriever.retrieve_similar_cases(query):
        print(f"\n    {Fore.WHITE}{c['title'][:60]}{Style.RESET_ALL}")
        kv("    Citation:", c["citation"],          Fore.CYAN)
        kv("    Court:",    c["court"],              Fore.CYAN)
        kv("    Year:",     str(c["year"]),          Fore.CYAN)
        kv("    Outcome:",  c["outcome"].upper(),    Fore.CYAN)
        kv("    Score:",    score_bar(c["similarity_score"]))

    # ── Step 4: BERT ────────────────────────────────────────────
    if use_bert:
        section("STEP 4 — BERT Semantic Embeddings (HF Pre-computed, 1024-dim)")
        from models.bert_embedder import BERTEmbedder
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        t0   = time.perf_counter()
        bert = BERTEmbedder(top_k=3, use_hf_embeddings=True)
        bert.build_index()
        t_bert = time.perf_counter() - t0

        kv("Index built in:", f"{t_bert:.1f}s (using pre-computed HF embeddings)", Fore.GREEN)
        kv("Embedding dim:",  "1024 (mxbai-embed-large-v1)", Fore.CYAN)

        info = bert.get_embedding_info(query)
        kv("Vector norm:", str(info["norm"]) + " (L2 normalized)", Fore.CYAN)
        kv("Dimensions:", str(info["dimensions"]), Fore.CYAN)

        # Semantic similarity demo
        print(f"\n  Semantic similarity demonstration:")
        q1 = "wife seeking divorce on grounds of cruelty"
        q2 = "husband domestic violence marriage dissolution"
        q3 = "murder accused section 302 criminal appeal"
        e1, e2, e3 = bert.encode_query(q1), bert.encode_query(q2), bert.encode_query(q3)
        s12 = cos_sim(e1, e2)[0][0]
        s13 = cos_sim(e1, e3)[0][0]
        print(f"  Q1: '{q1}'")
        print(f"  Q2: '{q2}'")
        kv("  Q1 ↔ Q2:", f"{s12:.3f} {Fore.GREEN}← HIGH similarity (same meaning){Style.RESET_ALL}")
        print(f"  Q3: '{q3}'")
        kv("  Q1 ↔ Q3:", f"{s13:.3f} {Fore.RED}← LOW similarity (different topic){Style.RESET_ALL}")

        print(f"\n  {Fore.YELLOW}BERT similar cases:{Style.RESET_ALL}")
        for c in bert.retrieve_similar_cases(query):
            print(f"\n    {Fore.WHITE}{c['title'][:60]}{Style.RESET_ALL}")
            kv("    Citation:", c["citation"],       Fore.CYAN)
            kv("    Score:",    score_bar(c["similarity_score"]))
    else:
        section("STEP 4 — BERT Embeddings [Skipped — use --fast to enable]")
        print(f"  {Fore.YELLOW}Run without --fast to include BERT semantic retrieval{Style.RESET_ALL}")

    # ── Step 5: Risk predictor ──────────────────────────────────
    section("STEP 5 — Risk Predictor (Random Forest + Gradient Boosting)")
    from models.risk_predictor import RiskPredictor

    t0   = time.perf_counter()
    pred = RiskPredictor()
    m    = pred.train()
    t_rp = time.perf_counter() - t0

    kv("Trained in:",      f"{t_rp:.1f}s on 1,414 real cases", Fore.GREEN)
    kv("Risk accuracy:",   f"{m['risk_accuracy']:.2%}", Fore.GREEN)
    kv("Outcome accuracy:", f"{m['outcome_accuracy']:.2%}", Fore.GREEN)

    for demo in DEMO_CASES[:3]:
        r = pred.predict(demo["text"])
        print(f"\n  {Fore.WHITE}{demo['label']}{Style.RESET_ALL}")
        rc = risk_color(r["risk_level"])
        kv("  Risk:",    f"{rc}{r['risk_level'].upper()}{Style.RESET_ALL} ({r['risk_confidence_pct']})", "")
        kv("  Outcome:", r["likely_outcome"].upper(), Fore.CYAN)
        for f in r["risk_factors"][:2]:
            bullet(f, Fore.RED)

    # ── Step 6: BiLSTM ─────────────────────────────────────────
    section("STEP 6 — BiLSTM Classifier (TensorFlow/Keras)")
    from models.bilstm_classifier import BiLSTMClassifier

    print(f"  Architecture: Input(512) → Embedding(30k,128) → BiLSTM(128) → Dense(64) → Softmax(7)")
    print(f"  Case types: criminal | civil | family | service | tax | corporate | constitutional")
    print(f"\n  {Fore.YELLOW}Using rule-based classifier (train=True for full model){Style.RESET_ALL}")

    clf = BiLSTMClassifier()
    for demo in DEMO_CASES:
        p = clf._rule_based_fallback(demo["text"])
        top3 = sorted(p["all_probabilities"].items(), key=lambda x: -x[1])[:3]
        print(f"\n  {Fore.WHITE}{demo['label']}{Style.RESET_ALL}")
        kv("  Predicted:", p["predicted_type"].upper(), Fore.TEAL)
        kv("  Confidence:", p["confidence_pct"],        Fore.CYAN)
        kv("  Top 3:",     str(top3),                    Fore.CYAN)

    # ── Step 7: Full engine ─────────────────────────────────────
    section("STEP 7 — Full Recommendation Engine (All Models Combined)")
    from models.recommendation_engine import RecommendationEngine

    engine = RecommendationEngine(use_bert=use_bert, use_bilstm=False)
    engine.tfidf          = retriever
    engine.risk_predictor = pred
    engine.classifier     = clf
    if use_bert:
        engine.bert = bert
    engine._models_initialized = True

    for demo in DEMO_CASES:
        print(f"\n  {'═'*58}")
        print(f"  {Fore.WHITE}{demo['label']}{Style.RESET_ALL}")
        print(f"  {'─'*58}")
        print(f"  {demo['text'][:130]}...")

        t0     = time.perf_counter()
        result = engine.analyze(demo["text"])
        rd     = engine.to_dict(result)
        t_eng  = time.perf_counter() - t0

        print(f"\n  {Fore.GREEN}── Analysis Results ──{Style.RESET_ALL}")
        kv("  Case type:",     rd["detected_case_type"].upper(), Fore.TEAL)
        rc = risk_color(rd["risk_level"])
        kv("  Risk level:",    f"{rc}{rd['risk_level'].upper()}{Style.RESET_ALL} ({rd['risk_confidence']:.0%})", "")
        kv("  Likely outcome:", rd["likely_outcome"].upper(), Fore.CYAN)
        kv("  Jurisdiction:",  rd["detected_jurisdiction"], Fore.CYAN)
        kv("  Processing:",    f"{rd['processing_time_sec']}s", Fore.CYAN)

        if rd["similar_cases"]:
            print(f"\n  {Fore.YELLOW}📋 Similar cases (real SC judgments):{Style.RESET_ALL}")
            for c in rd["similar_cases"][:2]:
                print(f"    [{score_bar(c['relevance_score'], 12)}]")
                print(f"     {c['title'][:55]}")
                print(f"     {Fore.CYAN}{c['citation']}{Style.RESET_ALL}  |  {c['outcome'].upper()}")

        if rd["recommended_resources"]:
            print(f"\n  {Fore.YELLOW}📚 Recommended resources:{Style.RESET_ALL}")
            for res in rd["recommended_resources"][:2]:
                t = f"[{res['resource_type'].upper()}]"
                print(f"    [{score_bar(res['relevance_score'], 12)}]")
                print(f"     {Fore.PURPLE}{t}{Style.RESET_ALL} {res['title'][:50]}")

        if rd["risk_factors"]:
            print(f"\n  {Fore.YELLOW}⚠  Risk factors:{Style.RESET_ALL}")
            for f in rd["risk_factors"][:2]:
                bullet(f, Fore.RED)

    # ── Final summary ──────────────────────────────────────────
    print(f"\n\n  {'═'*60}")
    print(f"  {Fore.GREEN}DEMO COMPLETE — Legum AI v2{Style.RESET_ALL}")
    print(f"  {'═'*60}")
    print(f"""
  {Fore.GREEN}✓{Style.RESET_ALL} HFDatasetLoader  — 1,414 real SC Pakistan judgments
  {Fore.GREEN}✓{Style.RESET_ALL} TextPreprocessor  — NLTK tokenization, stopwords, lemmatization
  {Fore.GREEN}✓{Style.RESET_ALL} TFIDFRetriever    — scikit-learn, 30k vocab, cosine similarity
  {Fore.GREEN}✓{Style.RESET_ALL} BERTEmbedder      — HF pre-computed 1024-dim mxbai embeddings
  {Fore.GREEN}✓{Style.RESET_ALL} BiLSTMClassifier  — TensorFlow/Keras, 512 seq len, 7 case types
  {Fore.GREEN}✓{Style.RESET_ALL} RiskPredictor     — Random Forest + Gradient Boosting, 5k features
  {Fore.GREEN}✓{Style.RESET_ALL} RecommendationEngine — Full pipeline integration

  {Fore.CYAN}Commands:
    python train.py                    # Train all models
    python api/app.py                  # Web UI at http://localhost:5000
    python utils/evaluator.py          # Generate presentation charts
    python tests/test_engine.py        # Run unit tests{Style.RESET_ALL}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legum AI v2 Demo")
    parser.add_argument("--fast",  action="store_true", help="Skip BERT (faster demo)")
    parser.add_argument("--query", type=str,            help="Custom case text to analyze")
    args = parser.parse_args()
    run_demo(use_bert=not args.fast, custom_query=args.query)
