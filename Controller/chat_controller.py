from fastapi import APIRouter
from pydantic import BaseModel
from Model.ai_engine import gerar

router = APIRouter()
class ChatRequest(BaseModel):
    prompt: str

@router.post("/chat")
def chat(request: ChatRequest):
    answer = gerar(request.prompt)

    return {
        "response": answer["text"],
        "latency": answer["latency"]
    }
