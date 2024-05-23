import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main1 import rag_chain

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/chatbot_api/")
async def chatbot_endpoint(query: Query):
    try:
        print(query.question)
        result = rag_chain.invoke(query.question)
        response = result.split("Answer: ")[1].strip()
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
