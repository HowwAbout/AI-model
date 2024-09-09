from dotenv import load_dotenv

from fastapi import FastAPI, Request

# Config
load_dotenv()

# Server Init
app = FastAPI()

@app.get("/")
def root():
    return {"message": "this is message and root"}