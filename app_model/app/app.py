from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model_handler import ModelHandler

app = FastAPI()

# Initialize the model handler 
model_handler = ModelHandler()

class Query(BaseModel):
    query: str

@app.post("/search")
async def predict(query: Query):
    try:
        top_results = model_handler.find_similar_products(query.query)
        return {"results": top_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
