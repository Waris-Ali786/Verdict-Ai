from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os

# Fix imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.recommendation_engine import RecommendationEngine
from utils.file_parser import FileParser

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
app = FastAPI(title="Legum AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
print("\nStarting Legum AI API (FastAPI)...")
engine = RecommendationEngine(use_bert=True, use_bilstm=True)
engine.initialize()
print("API ready.\n")

# ─────────────────────────────────────────
# Request Model
# ─────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    case_type: str | None = None


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Legum AI Recommendation Engine",
        "version": "1.0.0",
        "models": ["TF-IDF", "Sentence-BERT", "BiLSTM"]
    }


@app.post("/api/analyze-text")
def analyze_text(request: TextRequest):
    text = request.text.strip()

    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Text too short (min 20 chars)")

    try:
        result = engine.analyze(text)
        return {"success": True, **engine.to_dict(result)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        content = await file.read()

        FileParser.validate_file_size(content, max_mb=10)
        text = FileParser.parse(content, file.filename)

        if len(text) < 20:
            raise HTTPException(status_code=422, detail="No meaningful text extracted")

        result = engine.analyze(text)

        return {
            "success": True,
            "filename": file.filename,
            "text_length": len(text),
            **engine.to_dict(result)
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
