"""Minimal MLflow-compatible scoring server: GET /ping, POST /invocations."""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

MODEL_PATH = os.getenv("SCORING_MODEL_PATH", "scoring_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

app = FastAPI(title="Loan Scoring", version=str(bundle.get("version", "1.0.0")))


@app.get("/ping")
def ping():
    return PlainTextResponse("OK")


@app.post("/invocations")
async def invocations(request: Request):
    body = await request.json()
    if "dataframe_split" in body:
        split = body["dataframe_split"]
        X = pd.DataFrame(split["data"], columns=split["columns"])[features]
    elif "dataframe_records" in body:
        X = pd.DataFrame(body["dataframe_records"])[features]
    else:
        return JSONResponse({"error": "unsupported payload"}, status_code=400)
    proba = model.predict_proba(X)[:, 1].tolist()
    return JSONResponse(proba)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
