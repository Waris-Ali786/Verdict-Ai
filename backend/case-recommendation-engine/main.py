import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from colorama import Fore, Style, init
init(autoreset=True)


def print_banner():
    print(f"\n{Fore.GREEN}")
    print("  ██╗     ███████╗ ██████╗ ██╗   ██╗███╗   ███╗")
    print("  ██║     ██╔════╝██╔════╝ ██║   ██║████╗ ████║")
    print("  ██║     █████╗  ██║  ███╗██║   ██║██╔████╔██║")
    print("  ██║     ██╔══╝  ██║   ██║██║   ██║██║╚██╔╝██║")
    print("  ███████╗███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║")
    print(f"{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Legum AI v2 — Case Recommendation Engine{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}1,414 Supreme Court of Pakistan Judgments (HuggingFace){Style.RESET_ALL}\n")


def bar(score, w=18):
    f = int(score / 100 * w)
    b = "█" * f + "░" * (w - f)
    c = Fore.GREEN if score >= 70 else (Fore.YELLOW if score >= 40 else Fore.RED)
    return f"{c}{b}{Style.RESET_ALL} {score:.1f}%"


def rc(level):
    return {"high": Fore.RED, "medium": Fore.YELLOW, "low": Fore.GREEN}.get(level, Fore.WHITE)


def print_result(rd):
    r = rd
    print(f"\n{Fore.YELLOW}  ▸ CLASSIFICATION{Style.RESET_ALL}")
    print(f"  Type:         {Fore.TEAL}{r['detected_case_type'].upper()}{Style.RESET_ALL}"
          f"  ({r['case_type_confidence']:.0%} confidence)")
    print(f"  Jurisdiction: {r['detected_jurisdiction']}")

    print(f"\n{Fore.YELLOW}  ▸ RISK ASSESSMENT{Style.RESET_ALL}")
    col = rc(r["risk_level"])
    print(f"  Risk Level:     {col}{r['risk_level'].upper()}{Style.RESET_ALL}"
          f"  ({r['risk_confidence']:.0%})")
    print(f"  Likely Outcome: {Fore.CYAN}{r['likely_outcome'].upper()}{Style.RESET_ALL}"
          f"  ({r['outcome_confidence']:.0%})")
    for f in r.get("risk_factors", []):
        print(f"    {Fore.RED}•{Style.RESET_ALL} {f}")

    if r.get("key_legal_issues"):
        print(f"\n{Fore.YELLOW}  ▸ KEY LEGAL ISSUES{Style.RESET_ALL}")
        for kw in r["key_legal_issues"]:
            print(f"  {Fore.CYAN}•{Style.RESET_ALL} {kw}")

    if r.get("detected_statutes"):
        print(f"\n{Fore.YELLOW}  ▸ DETECTED STATUTES{Style.RESET_ALL}")
        for s in r["detected_statutes"]:
            print(f"  {Fore.PURPLE}•{Style.RESET_ALL} {s}")

    if r.get("similar_cases"):
        print(f"\n{Fore.YELLOW}  ▸ SIMILAR CASES (from 1,414 real SC judgments){Style.RESET_ALL}")
        for i, c in enumerate(r["similar_cases"], 1):
            print(f"\n  {i}. {Fore.WHITE}{c['title'][:60]}{Style.RESET_ALL}")
            print(f"     Citation:  {Fore.CYAN}{c['citation']}{Style.RESET_ALL}")
            print(f"     Court:     {c['court']} ({c['year']})")
            print(f"     Outcome:   {c['outcome'].upper()}  |  Risk: {c['risk_level'].upper()}")
            print(f"     Relevance: {bar(c['relevance_score'])}  via {c['retrieval_method']}")
            print(f"     URL:       {Fore.BLUE}{c['url']}{Style.RESET_ALL}")
            if c.get("facts_snippet"):
                print(f"     Facts:     {c['facts_snippet'][:120].replace(chr(10),' ')}...")

    if r.get("recommended_resources"):
        print(f"\n{Fore.YELLOW}  ▸ RECOMMENDED RESOURCES{Style.RESET_ALL}")
        for i, res in enumerate(r["recommended_resources"], 1):
            print(f"\n  {i}. [{res['resource_type'].upper()}] {Fore.WHITE}{res['title']}{Style.RESET_ALL}")
            print(f"     {res['description']}")
            print(f"     Relevance: {bar(res['relevance_score'])}")
            print(f"     URL: {Fore.BLUE}{res['url']}{Style.RESET_ALL}")

    if r.get("models_used"):
        print(f"\n{Fore.YELLOW}  ▸ MODELS USED{Style.RESET_ALL}")
        for m in r["models_used"]:
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {m}")
    print(f"\n  {Fore.CYAN}⏱  Processing: {r.get('processing_time_sec','?')}s{Style.RESET_ALL}")
    print(f"\n  {'─'*55}")


DEMO_CASES = [
    ("Criminal — Murder Section 302 PPC",
     "The accused Khalid Rehman charged under Section 302 PPC for murder of Shahid Mehmood "
     "in Karachi. FIR registered at Clifton Police Station. Sindh High Court upheld death "
     "sentence. Defense argues misidentification. Prosecution relies on eyewitness and ballistics."),
    ("Service — Civil Servant Wrongful Dismissal",
     "Civil servant employed with Pakistan Railways dismissed from service without show-cause "
     "notice violating principles of natural justice. Federal Service Tribunal dismissed appeal. "
     "Supreme Court petition seeks reinstatement with back pay and seniority restoration."),
    ("Family — Divorce and Custody",
     "Wife filed for Khul divorce in Family Court Lahore under Muslim Family Laws Ordinance 1961 "
     "citing domestic cruelty. Two minor children involved. Husband contesting divorce and seeking "
     "custody. Monthly maintenance and dower recovery under Guardian and Wards Act 1890 claimed."),
    ("Tax — Income Tax FBR Appeal",
     "Taxpayer challenged income tax assessment order under Income Tax Ordinance 2001. "
     "Commissioner Inland Revenue added unexplained income PKR 50 million without proper basis. "
     "Appellate tribunal dismissed. Questions of burden of proof and natural justice raised."),
]


def show_dataset_stats():
    print_banner()
    print(f"  {Fore.CYAN}Loading HuggingFace dataset statistics...{Style.RESET_ALL}\n")
    from data.hf_dataset_loader import HFDatasetLoader
    import numpy as np
    loader = HFDatasetLoader()
    df     = loader.load()
    loader.summary()
    emb = loader.get_embeddings_matrix()
    print(f"\n  {Fore.YELLOW}Embedding Matrix:{Style.RESET_ALL}")
    print(f"  Shape:   {emb.shape}  (1414 cases × 1024 dims)")
    print(f"  dtype:   {emb.dtype}")
    print(f"  Model:   mixedbread-ai/mxbai-embed-large-v1")
    print(f"  Avg norm: {np.linalg.norm(emb, axis=1).mean():.4f} (should be ~1.0)")


def main():
    parser = argparse.ArgumentParser(description="Legum AI v2 CLI")
    parser.add_argument("--text",    type=str)
    parser.add_argument("--file",    type=str)
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--json",    action="store_true")
    parser.add_argument("--fast",    action="store_true", help="Skip BERT")
    args = parser.parse_args()

    if not any([args.text, args.file, args.demo, args.dataset]):
        parser.print_help()
        sys.exit(1)

    if args.dataset:
        show_dataset_stats()
        return

    print_banner()

    from models.recommendation_engine import RecommendationEngine
    engine = RecommendationEngine(use_bert=not args.fast, use_bilstm=False)
    engine.initialize()

    cases = []
    if args.demo:
        cases = list(DEMO_CASES)
    elif args.file:
        from utils.file_parser import FileParser
        print(f"  Reading: {args.file}")
        with open(args.file, "rb") as f:
            fb = f.read()
        text = FileParser.parse(fb, args.file)
        print(f"  Extracted {len(text):,} chars.\n")
        cases = [(os.path.basename(args.file), text)]
    elif args.text:
        cases = [("Custom query", args.text)]

    for label, text in cases:
        print(f"\n  {'═'*58}")
        print(f"  {Fore.CYAN}{label}{Style.RESET_ALL}")
        print(f"  {'═'*58}")
        print(f"  {text[:100]}{'...' if len(text)>100 else ''}\n")
        result = engine.analyze(text)
        rd     = engine.to_dict(result)
        if args.json:
            print(json.dumps(rd, indent=2, ensure_ascii=False))
        else:
            print_result(rd)


if __name__ == "__main__":
    main()
