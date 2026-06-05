from fastapi import APIRouter
from Model.ai_engine import gerar

router = APIRouter()

@router.post("/chat")
def chat(prompt: str):
    answer = gerar(prompt)

    return {
        "response": answer
    }