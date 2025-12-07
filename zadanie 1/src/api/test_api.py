from fastapi import FastAPI

app = FastAPI(title="Ecommerce ML API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Ecommerce ML API работает!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict():
    return {"prediction": 0.85, "model": "test_model"}