# app.py

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from query_index import search_index

import os
import sys
import argparse
import requests
import json

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(
    title="OpenAdonAI Archetype Oracle",
    description="Local RAG API over Ishmael's Obsidian Archetype vault.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = -1.0


class SearchResult(BaseModel):
    rank: int
    score: float
    file_path: str
    chunk_index: int
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    top_k: int
    min_score: float
    results: List[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok", "message": "Archetype Oracle is live."}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    POST /search
    Body: { "query": "...", "top_k": 5, "min_score": -1.0 }

    Returns top_k matching chunks from your Archetype index.
    """
    # Make sure we always have a list, even if search_index misbehaves
    results_raw = search_index(req.query, top_k=req.top_k, min_score=req.min_score) or []

    # Convert raw dicts from search_index into Pydantic-friendly shapes
    results: List[SearchResult] = []
    for r in results_raw:
        results.append(
            SearchResult(
                rank=int(r["rank"]),
                score=float(r["score"]),
                file_path=str(r.get("file_path", "")),
                chunk_index=int(r.get("chunk_index", -1)),
                text=r["text"],
                metadata=r["metadata"],
            )
        )

    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
        results=results,
    )
