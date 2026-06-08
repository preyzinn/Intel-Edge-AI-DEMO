from fastapi import FastAPI
from Controller.chat_controller import router as chat_router

app = FastAPI()

app.include_router(chat_router)

@app.get("/")
def root():
    return {"message": "backend ta rodando!!! 🚀"}
