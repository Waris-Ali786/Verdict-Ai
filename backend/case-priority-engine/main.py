from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from config          import COHERE_API_KEY
from pdf_extractor   import extract_and_clean
from signal_extractor import extract_signals_with_ai, fallback_extract
from scorer          import calculate_priority_score, get_tag

app = FastAPI(title="Case Priority Tagger API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # only allow Streamlit
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Case Priority Tagger API is running. Go to /docs for API documentation."}

@app.post("/process")
async def process_cases(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed.")

    results = []

    for f in files:
        if not f.filename.endswith(".pdf"):
            continue

        pdf_bytes = await f.read()

        // Step 1: extract text
        try:
            text = extract_and_clean(pdf_bytes)
        except Exception as e:
            results.append({"filename": f.filename, "error": f"Could not read PDF: {e}"})
            continue

        if len(text) < 50:
            results.append({"filename": f.filename, "error": "No extractable text. May be a scanned PDF."})
            continue

        // Step 2: extract signals
        try:
            signals = extract_signals_with_ai(text)
        except Exception:
            signals = fallback_extract(text)

        // Step 3: score
        score, breakdown = calculate_priority_score(signals)
        tag = get_tag(score)

        // Step 4: resolve title
        title = (
            signals.get("case_title") or
            signals.get("title")      or
            signals.get("case_name")  or
            f.filename.replace(".pdf", "")
        )

        results.append({
            "filename": f.filename,
            "title":    title,
            "signals":  signals,
            "score":    score,
            "tag":      tag,
            "breakdown": breakdown,
        })

    // Sort by score
    results = [r for r in results if "error" not in r]
    results.sort(key=lambda x: x["score"], reverse=True)

    return {"cases": results, "total": len(results)}
