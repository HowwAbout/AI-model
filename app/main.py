from dotenv import load_dotenv

from fastapi import FastAPI, Request

from .routers import dating_generation_router

# Config
load_dotenv()

# Server Init
app = FastAPI()

# Router
app.include_router(dating_generation_router.router)

@app.get("/")
def root():
    return {"message": "this is message and root"}