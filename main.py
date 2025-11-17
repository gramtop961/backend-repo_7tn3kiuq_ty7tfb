import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from database import db, create_document, get_documents
from schemas import User

# External services
from openai import OpenAI
from google.cloud import vision

app = FastAPI(title="Mathimatikos.xyz API", description="AI-powered math learning platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------
class OCRRequest(BaseModel):
    image_url: Optional[str] = None
    language: str = Field("en", description="en or el (Greek)")

class SolveRequest(BaseModel):
    latex: str
    language: str = Field("en", description="en or el (Greek)")
    grade: Optional[str] = Field(None, description="Target grade level, e.g., middle, high, university")

class GenerateRequest(BaseModel):
    topic: str
    count: int = 5
    language: str = Field("en", description="en or el (Greek)")
    difficulty: str = Field("mixed", description="easy|medium|hard|mixed")

# ---------- Helpers ----------

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return OpenAI(api_key=api_key)


def get_vision_client() -> vision.ImageAnnotatorClient:
    # Assumes GOOGLE_APPLICATION_CREDENTIALS env var is set
    try:
        return vision.ImageAnnotatorClient()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Vision not configured: {str(e)[:120]}")


def language_prefix(language: str) -> str:
    return "[English]" if language.lower().startswith("en") else "[Greek]"


SYSTEM_SOLVER = (
    "You are Mathimatikos.xyz, an expert math tutor. "
    "Solve the problem step-by-step with precise reasoning, show LaTeX for formulas, and explain why each step is valid. "
    "Prefer concise, clear pedagogy tailored to the student's level."
)

SYSTEM_GENERATOR = (
    "You are Mathimatikos.xyz, a problem generator for teachers. "
    "Produce diverse exercises with increasing variety. Provide solutions and short pedagogical notes. Use LaTeX for math."
)

# ---------- Routes ----------

@app.get("/")
def read_root():
    return {"message": "Mathimatikos.xyz backend is running"}


@app.post("/api/ocr")
async def ocr_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    language: str = Form("en")
):
    """Extract text via Google Vision and return LaTeX guess using OpenAI cleaning/parsing."""
    # 1) OCR via Vision
    client = get_vision_client()
    image = None
    if file is not None:
        content = await file.read()
        image = vision.Image(content=content)
    elif image_url:
        image = vision.Image()
        image.source.image_uri = image_url
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or image_url")

    response = client.text_detection(image=image)
    if response.error.message:
        raise HTTPException(status_code=500, detail=f"Vision error: {response.error.message}")

    raw_text = response.full_text_annotation.text if response.full_text_annotation else ""

    # 2) Convert to LaTeX with OpenAI
    oai = get_openai_client()
    prefix = language_prefix(language)

    prompt = (
        f"{prefix} Convert the following OCR text of a math problem into clean LaTeX. "
        "If the problem contains multiple parts, structure them clearly. "
        "Return only LaTeX between \n```latex\n and \n```\n.\n\n" 
        f"OCR text:\n{raw_text}"
    )

    try:
        completion = oai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_SOLVER},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)[:200]}")

    return {"text": raw_text, "latex": content}


@app.post("/api/solve")
async def solve_math(req: SolveRequest):
    oai = get_openai_client()
    prefix = language_prefix(req.language)

    user_msg = (
        f"{prefix} Solve the following problem, showing detailed reasoning and steps. "
        "Explain the principles behind each step. Use LaTeX for math.\n\n"
        f"Problem (LaTeX):\n{req.latex}"
    )

    try:
        completion = oai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_SOLVER},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)[:200]}")

    # Optionally store queries for analytics
    try:
        create_document("solve_log", {
            "latex": req.latex,
            "language": req.language,
            "grade": req.grade,
            "output": content,
        })
    except Exception:
        pass

    return {"solution": content}


@app.post("/api/generate")
async def generate_exercises(req: GenerateRequest):
    oai = get_openai_client()
    prefix = language_prefix(req.language)

    user_msg = (
        f"{prefix} Generate {req.count} math exercises on the topic '{req.topic}' (difficulty: {req.difficulty}). "
        "Provide each as: Problem (LaTeX), Solution (with steps), and a short teaching note (1-2 lines)."
    )

    try:
        completion = oai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_GENERATOR},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)[:200]}")

    try:
        create_document("generate_log", {
            "topic": req.topic,
            "count": req.count,
            "difficulty": req.difficulty,
            "language": req.language,
            "output": content,
        })
    except Exception:
        pass

    return {"exercises": content}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
