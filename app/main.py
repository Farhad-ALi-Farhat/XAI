from fastapi import FastAPI
from app.lime_api import lime_router  # hypothetical router

app = FastAPI()
app.include_router(lime_router)
