from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from Model.ai_engine import gerar, instalar_modelo, listar_modelos, obter_hardware


router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str
    model_id: str = "ollama-llama3"
    inference_device: str | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)


@router.get("/models")
def models():
    return {"models": listar_modelos()}


@router.get("/hardware")
def hardware():
    return obter_hardware()


@router.post("/models/{model_id}/install")
def install_model(model_id: str):
    try:
        return {"model": instalar_modelo(model_id)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = gerar(
            request.prompt,
            request.model_id,
            request.inference_device,
            request.messages,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "response": answer["text"],
        "latency": answer["latency"],
        "generated_tokens": answer["generated_tokens"],
        "tokens_per_second": answer["tokens_per_second"],
        "model_id": answer["model_id"],
        "family_id": answer["family_id"],
        "family_name": answer["family_name"],
        "backend": answer["backend"],
        "optimized": answer["optimized"],
        "inference_device": answer["inference_device"],
        "hardware_metrics": answer["hardware_metrics"],
    }
