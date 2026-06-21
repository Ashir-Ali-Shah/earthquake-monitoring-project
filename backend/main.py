import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_num_threads(1)
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
import json
import pickle
import traceback
from collections import Counter
import time
import threading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

warnings.filterwarnings('ignore')

try:
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    # import faiss  # Removed strictly to prevent memory corruption (free(): invalid pointer) with PyTorch OpenMP!
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    print("SentenceTransformers not available. Vector search disabled.")

try:
    import spacy
    nlp_model = None # spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = False
except:
    SPACY_AVAILABLE = False
    nlp_model = None
    print("spaCy not available. Entity extraction disabled.")

# ------------------- Config -------------------
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

try:
    from pinecone import Pinecone, ServerlessSpec
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    else:
        pc = None
except ImportError:
    pc = None

app = FastAPI(
    title="USGS Earthquake Intelligence System",
    description="Real-time earthquake analysis with RAG, ML, and Advanced LSTM forecasting",
    version="3.4.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Global State -------------------
earthquake_df: Optional[pd.DataFrame] = None

vector_store = None
simple_rag = None

# ------------------- Pydantic Models -------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class RAGQueryRequest(BaseModel):
    question: str



# ------------------- Helper Functions -------------------



def fetch_earthquakes():
    """Fetch earthquake data from USGS API"""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        "endtime": datetime.now().strftime("%Y-%m-%d"),
        "minmagnitude": 2.5,
        "limit": 20000,
        "orderby": "time"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        features = r.json().get("features", [])
        print(f"Fetched {len(features)} earthquakes from USGS")
        return features
    except Exception as e:
        print(f"USGS fetch failed: {e}")
        return []

def process_data(raw):
    """Process raw earthquake data into DataFrame"""
    if not raw:
        return None
    
    records = []
    for f in raw:
        p = f["properties"]
        c = f["geometry"]["coordinates"]
        
        # Calculate time ago
        event_time = pd.to_datetime(p.get("time"), unit='ms', utc=True) if p.get("time") else datetime.now(timezone.utc)
        
        # Format time for display
        time_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
        
        records.append({
            "id": f["id"],
            "magnitude": p.get("mag", 0.0),
            "place": p.get("place", "Unknown"),
            "time": event_time, # Keep as datetime object for sorting/math
            "timestamp": event_time, # Alias
            "timestamp_str": time_str,
            "latitude": c[1] if len(c) > 1 else 0.0,
            "longitude": c[0] if len(c) > 0 else 0.0,
            "depth": c[2] if len(c) > 2 else 0.0,
            "tsunami": p.get("tsunami", 0),
            "title": p.get("title", "")
        })
    
    df = pd.DataFrame(records)
    
    # Categorization helpers
    df['magnitude_category'] = pd.cut(
        df['magnitude'], 
        bins=[0, 5, 6, 7, 8, 10], 
        labels=['Minor', 'Moderate', 'Strong', 'Major', 'Great']
    )
    
    df['depth_category'] = pd.cut(
        df['depth'], 
        bins=[0, 70, 300, 1000], 
        labels=['Shallow', 'Intermediate', 'Deep']
    )
    
    df['location_clean'] = df['place'].str.extract(r'of (.+)$', expand=False).fillna(df['place'])
    
    return df



# ------------------- Vector Store & RAG -------------------
class VectorStore:
    """Vector store for semantic search of earthquake events"""
    
    def __init__(self):
        self.model = None
        self.index = None
    
    def build(self, df: pd.DataFrame):
        """Build Pinecone index from earthquake data"""
        if not VECTOR_SEARCH_AVAILABLE or pc is None or pinecone_index is None:
            return

        try:
            print("[DEBUG] Loading SentenceTransformer...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[DEBUG] SentenceTransformer loaded.")
            
            # Fast startup: skip re-upserting if vector store already has records
            print("[DEBUG] Fetching Pinecone stats...")
            stats = pinecone_index.describe_index_stats()
            print(f"[DEBUG] Pinecone stats fetched: {stats}")
            if stats.get('total_vector_count', 0) > 0:
                print(f"Pinecone index already populated with {stats.get('total_vector_count')} records. Skipping upsert.")
                self.index = True
                return
            
            print(f"Starting Pinecone upsert for {len(df)} records...")
            batch_size = 200
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                texts = []
                metas = []
                ids = []
                for idx, r in batch_df.iterrows():
                    text = f"Magnitude {r['magnitude']} earthquake {r['place']} at depth {r['depth']}km on {r['timestamp_str']}"
                    texts.append(text)
                    ids.append(str(r.get('id', str(idx))))
                    metas.append({
                        "magnitude": float(r['magnitude']) if pd.notna(r['magnitude']) else 0.0,
                        "place": str(r['place']),
                        "timestamp": str(r['timestamp_str']),
                        "latitude": float(r['latitude']) if pd.notna(r['latitude']) else 0.0,
                        "longitude": float(r['longitude']) if pd.notna(r['longitude']) else 0.0,
                        "depth": float(r['depth']) if pd.notna(r['depth']) else 0.0,
                        "title": str(r.get('title', ''))
                    })
                
                # Batch encode
                embeddings = self.model.encode(texts).tolist()
                
                vectors = []
                for j in range(len(ids)):
                    vectors.append({"id": ids[j], "values": embeddings[j], "metadata": metas[j]})
                    
                pinecone_index.upsert(vectors=vectors)
            
            self.index = True
            
            print(f"Vector store built via Pinecone with {len(df)} entries")
        except Exception as e:
            print(f"Vector store build failed: {e}")
    
    def search(self, query: str, k: int = 5):
        """Search for similar earthquake events"""
        if not getattr(self, "index", None) or not self.model or pinecone_index is None:
            return []
        
        try:
            q_vec = self.model.encode(query).tolist()
            
            res = pinecone_index.query(vector=q_vec, top_k=k, include_metadata=True)
            
            results = []
            for match in res.get('matches', []):
                meta = match.get('metadata', {})
                meta['id'] = match.get('id')
                meta['similarity_score'] = match.get('score')
                results.append(meta)
            
            return results
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

class SimpleRAG:
    """RAG system for earthquake question answering"""
    
    def __init__(self):
        self.model = None
        self.index = None
    
    def build(self, df: pd.DataFrame):
        """Build RAG index from earthquake data"""
        if not VECTOR_SEARCH_AVAILABLE or pc is None or pinecone_index is None:
            return

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Since VectorStore already upserts all vectors with metadata,
            # we can just use the Pinecone index for queries directly.
            self.index = True
            
            print("RAG index ready via Pinecone")
        except Exception as e:
            print(f"RAG build failed: {e}")
    
    def query_stream(self, question: str, k: int = 5):
        """Query RAG system with a question and stream response"""
        if not getattr(self, "index", None) or not self.model or pinecone_index is None:
            yield f"data: {json.dumps({'error': 'RAG system not initialized'})}\n\n"
            return
            
        t_start_precompute = time.time()
        

        # Initialize metrics for update in finally block
        precompute_latency = 0.0
        ttft = 0.0
        tpot = 0.0
        tps = 0.0
        full_answer = ""
        sources = []
        
        try:
            q_vec = self.model.encode(question).tolist()
            
            res = pinecone_index.query(vector=q_vec, top_k=k, include_metadata=True)
            for match in res.get('matches', []):
                meta = match.get('metadata', {})
                meta['id'] = match.get('id')
                sources.append(meta)
            
            t_end_precompute = time.time()
            precompute_latency = t_end_precompute - t_start_precompute
            
            # Build context for LLM
            context = "\n".join([
                f"{i+1}. M{s.get('magnitude', '')} {s.get('place', '')} ({s.get('timestamp', '')})" 
                for i, s in enumerate(sources[:3])
            ])
            
            prompt = f"Based on earthquake data, answer this question: {question}\n\nRelevant events:\n{context}\n\nProvide a concise answer:"
            
            # Send initial metadata
            initial_data = {
                "type": "metadata",
                "sources": sources[:3]
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # Variables for metrics tracking
            total_tokens = 0
            first_token_time = None
            last_token_time = None
            token_intervals = []
            
            # Try to call Groq API with streaming
            try:
                if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
                    payload = {
                        "model": GROQ_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 512,
                        "stream": True
                    }
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}", 
                        "Content-Type": "application/json"
                    }
                    
                    with requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=20, stream=True) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    data_str = line[len("data: "):]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        content = data_json['choices'][0]['delta'].get('content', '')
                                        if content:
                                            current_time = time.time()
                                            if first_token_time is None:
                                                first_token_time = current_time
                                                ttft = first_token_time - t_start_precompute
                                            else:
                                                token_intervals.append(current_time - last_token_time)
                                            
                                            last_token_time = current_time
                                            total_tokens += 1
                                            full_answer += content
                                            
                                            chunk_data = {"type": "chunk", "content": content}
                                            yield f"data: {json.dumps(chunk_data)}\n\n"
                                    except Exception:
                                        pass
                else:
                    answer = self._generate_simple_answer(question, sources)
                    full_answer = answer
                    yield f"data: {json.dumps({'type': 'chunk', 'content': answer})}\n\n"
            except Exception as e:
                print(f"LLM streaming call failed: {e}")
                answer = self._generate_simple_answer(question, sources)
                full_answer = answer
                yield f"data: {json.dumps({'type': 'chunk', 'content': answer})}\n\n"
            
            # Finalize metrics
            if token_intervals:
                tpot = sum(token_intervals) / len(token_intervals)
            if first_token_time and last_token_time and last_token_time > first_token_time:
                tps = total_tokens / (last_token_time - first_token_time)
                
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"RAG query failed: {e}")
            yield f"data: {json.dumps({'error': 'Failed to process query'})}\n\n"

    
    def _generate_simple_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate a simple answer without LLM"""
        if not sources:
            return "No relevant earthquake data found for your question."
        
        # Extract statistics
        magnitudes = [s.get('magnitude', 0) for s in sources]
        places = [s.get('place', '') for s in sources]
        
        avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0
        max_mag = max(magnitudes) if magnitudes else 0
        
        # Count most common location
        location_counter = Counter(places)
        most_common_loc = location_counter.most_common(1)[0][0] if location_counter else "various locations"
        
        answer = f"Based on {len(sources)} relevant earthquake events: "
        answer += f"The average magnitude is {avg_mag:.1f}, with a maximum of {max_mag:.1f}. "
        answer += f"Most activity occurred in {most_common_loc}."
        
        return answer

# Initialize global instances
vector_store = VectorStore()
simple_rag = SimpleRAG()

pinecone_index = None

# ------------------- Initialization -------------------
def init_system():
    """Initialize the entire system"""
    global earthquake_df
    global pinecone_index
    
    print("Initializing system...")
    
    if pc is not None:
        try:
            # Note: For Pinecone v3, pc.list_indexes().names() returns the list of names
            active_indexes = pc.list_indexes().names()
            if "earthquake-index" not in active_indexes:
                print("Creating Pinecone index 'earthquake-index'...")
                pc.create_index(
                    name="earthquake-index", 
                    dimension=384, 
                    metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            pinecone_index = pc.Index("earthquake-index")
            print("Pinecone index initialized.")
        except Exception as e:
            print(f"Failed to initialize Pinecone index: {e}")
            pinecone_index = None
    
    # Fetch earthquake data
    raw = fetch_earthquakes()
    df = process_data(raw)
    
    if df is None or len(df) == 0:
        print("Failed to load earthquake data")
        return False
    
    earthquake_df = df
    print(f"Loaded {len(df)} earthquakes")
    
    # Build vector store FIRST (to prevent PyTorch/TF thread deadlock)
    vector_store.build(df)
    
    # Build RAG system
    simple_rag.build(df)

    print("System ready!")
    return True

@app.on_event("startup")
async def startup():
    """Startup event handler"""
    init_system()

# ------------------- Routes -------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "USGS Earthquake Intelligence System API",
        "version": "3.4.1",
        "status": "operational",
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/stats",
            "recent": "/api/earthquakes/recent",
            "hotspots": "/api/hotspots",
            "search": "/api/search",
            "rag": "/api/rag/query",
            "predict_lstm": "/api/predict/lstm",
            "refresh": "/api/refresh"
        }
    }

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": earthquake_df is not None,
        "total_earthquakes": len(earthquake_df) if earthquake_df is not None else 0,
        "rag_ready": simple_rag.index is not None,
        "vector_search_ready": vector_store.index is not None
    }

@app.post("/api/refresh")
async def refresh():
    """Refresh system data"""
    try:
        success = init_system()
        return {
            "status": "success" if success else "failed",
            "timestamp": datetime.now().isoformat(),
            "earthquakes_loaded": len(earthquake_df) if earthquake_df is not None else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")

@app.get("/api/stats")
async def stats():
    """Get earthquake statistics"""
    if earthquake_df is None or len(earthquake_df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = earthquake_df.copy()
    total = len(df)
    
    # Category distribution
    cat_dist = df['magnitude_category'].value_counts().to_dict()
    depth_dist = df['depth_category'].value_counts().to_dict()
    
    # Calculate statistics
    mag_mean = df['magnitude'].mean() if not df['magnitude'].isna().all() else 0
    mag_min = df['magnitude'].min() if not df['magnitude'].isna().all() else 0
    mag_max = df['magnitude'].max() if not df['magnitude'].isna().all() else 0
    
    high_risk_count = int((df['magnitude'] >= 6.0).sum())
    tsunami_count = int(df['tsunami'].sum())
    unique_regions = int(df['location_clean'].nunique())
    
    return {
        "total_count": int(total),
        "category_distribution": {str(k): int(v) for k, v in cat_dist.items()},
        "depth_distribution": {str(k): int(v) for k, v in depth_dist.items()},
        "magnitude_stats": {
            "mean": round(float(mag_mean), 2),
            "min": round(float(mag_min), 2),
            "max": round(float(mag_max), 2)
        },
        "risk_metrics": {
            "high_risk_count": high_risk_count,
            "tsunami_warnings": tsunami_count
        },
        "geographic_metrics": {
            "unique_regions": unique_regions
        }
    }

@app.get("/api/earthquakes/recent")
async def recent(limit: int = Query(default=20, ge=1, le=100)):
    """Get recent earthquakes"""
    if earthquake_df is None:
        return {"earthquakes": []}
    
    df = earthquake_df.sort_values('time', ascending=False).head(limit)
    
    # Convert to records with proper types
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row['id']),
            "magnitude": float(row['magnitude']),
            "place": str(row['place']),
            "timestamp": str(row['timestamp_str']),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "depth": float(row['depth']),
            "magnitude_category": str(row.get('magnitude_category', 'Unknown')),
            "depth_category": str(row.get('depth_category', 'Unknown'))
        })
    
    return {"earthquakes": records}

@app.get("/api/hotspots")
async def hotspots(limit: int = Query(default=10, ge=1, le=50)):
    """Get earthquake hotspots"""
    if earthquake_df is None or len(earthquake_df) == 0:
        return {"hotspots": []}
    
    df = earthquake_df.copy()
    top_regions = df['location_clean'].value_counts().head(limit).index
    
    results = []
    for region in top_regions:
        sub = df[df['location_clean'] == region]
        latest = sub.sort_values('time', ascending=False).iloc[0]
        
        results.append({
            "region": str(region),
            "count": int(len(sub)),
            "avg_magnitude": round(float(sub['magnitude'].mean()), 2),
            "max_magnitude": round(float(sub['magnitude'].max()), 2),
            "avg_depth": round(float(sub['depth'].mean()), 1),
            "latest_earthquake": {
                "magnitude": round(float(latest['magnitude']), 2),
                "depth": round(float(latest['depth']), 1),
                "time": str(latest['timestamp_str']),
                "latitude": round(float(latest['latitude']), 4),
                "longitude": round(float(latest['longitude']), 4)
            }
        })
    
    return {"hotspots": results}

@app.post("/api/search")
def search(req: SearchRequest):
    """Semantic search for earthquakes"""
    if earthquake_df is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    # Extract entities if available
    entities = []
    if SPACY_AVAILABLE and nlp_model:
        try:
            doc = nlp_model(req.query)
            entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
        except:
            pass
    
    # Perform search
    results = vector_store.search(req.query, req.top_k)
    
    return {
        "query": req.query,
        "extracted_entities": entities,
        "results": results,
        "count": len(results)
    }

@app.post("/api/rag/query")
def rag(req: RAGQueryRequest):
    """RAG-based question answering (Streaming)"""
    # Extract entities if available
    entities = []
    if SPACY_AVAILABLE and nlp_model:
        try:
            doc = nlp_model(req.question)
            entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
        except:
            pass
    
    def event_stream():
        yield f"data: {json.dumps({'type': 'entities', 'extracted_entities': entities})}\n\n"
        yield from simple_rag.query_stream(req.question)
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")



# ------------------- Additional Utility Endpoints -------------------

@app.get("/api/earthquakes/by-region")
async def earthquakes_by_region(region: str, limit: int = Query(default=50, ge=1, le=200)):
    """Get earthquakes for a specific region"""
    if earthquake_df is None:
        return {"earthquakes": []}
    
    df = earthquake_df[earthquake_df['place'].str.contains(region, case=False, na=False)]
    df = df.sort_values('time', ascending=False).head(limit)
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row['id']),
            "magnitude": float(row['magnitude']),
            "place": str(row['place']),
            "timestamp": str(row['timestamp_str']),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "depth": float(row['depth'])
        })
    
    return {
        "region": region,
        "earthquakes": records,
        "count": len(records)
    }

@app.get("/api/earthquakes/magnitude-range")
async def earthquakes_by_magnitude(
    min_mag: float = Query(default=4.0, ge=0, le=10),
    max_mag: float = Query(default=10.0, ge=0, le=10),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Get earthquakes within magnitude range"""
    if earthquake_df is None:
        return {"earthquakes": []}
    
    df = earthquake_df[
        (earthquake_df['magnitude'] >= min_mag) & 
        (earthquake_df['magnitude'] <= max_mag)
    ]
    df = df.sort_values('magnitude', ascending=False).head(limit)
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row['id']),
            "magnitude": float(row['magnitude']),
            "place": str(row['place']),
            "timestamp": str(row['timestamp_str']),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "depth": float(row['depth'])
        })
    
    return {
        "min_magnitude": min_mag,
        "max_magnitude": max_mag,
        "earthquakes": records,
        "count": len(records)
    }

@app.get("/api/analytics/timeline")
async def timeline():
    """Get earthquake timeline data"""
    if earthquake_df is None:
        return {"timeline": []}
    
    df = earthquake_df.copy()
    df['date'] = df['timestamp'].dt.date
    
    timeline = df.groupby('date').agg({
        'magnitude': ['count', 'mean', 'max']
    }).reset_index()
    
    timeline.columns = ['date', 'count', 'avg_magnitude', 'max_magnitude']
    timeline['date'] = timeline['date'].astype(str)
    
    return {
        "timeline": timeline.to_dict('records')
    }

# ------------------- Error Handlers -------------------

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "details": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting USGS Earthquake Intelligence System API...")
    print("Documentation available at: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)