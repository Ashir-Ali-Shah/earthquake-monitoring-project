from fastapi import FastAPI, HTTPException, Query
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
import os
import traceback
from collections import Counter

# Updated Import to match your training script
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ------------------- Model Imports -------------------

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not available. LSTM forecasting disabled.")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    print("SentenceTransformers or FAISS not available. Vector search disabled.")

try:
    import spacy
    nlp_model = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    nlp_model = None
    print("spaCy not available. Entity extraction disabled.")

# ------------------- Config -------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-70b-versatile"

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
lstm_model = None
lstm_scaler = None
lstm_features = None
lstm_sequence_data = None

# Updated Sequence Length based on your script
LSTM_SEQ_LENGTH = 40 

vector_store = None
simple_rag = None

# ------------------- Pydantic Models -------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class RAGQueryRequest(BaseModel):
    question: str

class LSTMPredictionRequest(BaseModel):
    num_predictions: int = Field(default=1, ge=1, le=10)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    magnitude: Optional[float] = None
    delta_t: Optional[float] = None
    log_cum_energy_50: Optional[float] = None

# ------------------- Helper Functions -------------------

def rebuild_lstm_model(input_shape):
    """
    Rebuild LSTM model matching the integrated training script architecture.
    Structure: LSTM(128) -> Dropout(0.3) -> Dense(1)
    """
    model = Sequential()
    # input_shape = (seq_length, len(features))
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def initialize_lstm_model():
    """Initialize LSTM model and Scaler with Shape Mismatch Protection"""
    global lstm_model, lstm_scaler, lstm_features
    
    if not KERAS_AVAILABLE:
        print("Keras not available")
        return False
    
    try:
        # Load Scaler and Features
        if os.path.exists('scaler.pkl') and os.path.exists('features.json'):
            with open('scaler.pkl', 'rb') as f:
                lstm_scaler = pickle.load(f)
            with open('features.json', 'r') as f:
                lstm_features = json.load(f)
            
            # Feature list from your script
            feature_count = 6 
            
            # Try to load model
            if os.path.exists('earthquake_model.keras'):
                try:
                    lstm_model = keras.models.load_model('earthquake_model.keras', compile=False)
                    
                    # --- CRITICAL FIX FOR SHAPE MISMATCH ---
                    # Check if the loaded model expects 40 steps or 10 steps
                    try:
                        input_shape = lstm_model.input_shape
                        # input_shape is usually (None, seq_len, features)
                        if input_shape[1] != LSTM_SEQ_LENGTH:
                            print(f"Old model detected (Seq Length {input_shape[1]}). Expected {LSTM_SEQ_LENGTH}.")
                            print("Deleting old model to force retrain...")
                            del lstm_model
                            lstm_model = None
                            os.remove('earthquake_model.keras')
                            # Raise error to jump to the fresh setup block
                            raise ValueError("Model shape mismatch forced retrain")
                    except AttributeError:
                        pass # If we can't check shape, assume it might fail later
                    
                    if lstm_model:
                        lstm_model.compile(optimizer='adam', loss='mse')
                        print("LSTM model loaded successfully")
                        return True
                        
                except Exception as e:
                    print(f"Model load failed ({e}). Rebuilding...")
                    if os.path.exists('earthquake_model.keras'):
                        os.remove('earthquake_model.keras')
            
            # Create new model architecture if load failed or file deleted
            lstm_model = rebuild_lstm_model((LSTM_SEQ_LENGTH, feature_count))
            print("New LSTM model architecture created")
            return True
    except Exception as e:
        print(f"LSTM initialization warning: {e}")
    
    # Create fresh setup if anything above fails
    try:
        print("Creating fresh LSTM setup...")
        # Features defined in your script
        lstm_features = {
            'feature_names': ['magnitude', 'depth', 'latitude', 'longitude', 'delta_t', 'log_cum_energy_50']
        }
        # Using MinMaxScaler as per your script
        lstm_scaler = MinMaxScaler()
        
        # Fit scaler with dummy data to initialize (will be overwritten by real data)
        X_dummy = np.random.rand(100, 6)
        lstm_scaler.fit(X_dummy)
        
        # Create model
        lstm_model = rebuild_lstm_model((LSTM_SEQ_LENGTH, 6))
        
        # Save components
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(lstm_scaler, f)
        with open('features.json', 'w') as f:
            json.dump(lstm_features, f)
        
        print("Minimal LSTM setup created")
        return True
    except Exception as e:
        print(f"Failed to create LSTM setup: {e}")
        return False

def prepare_lstm_sequence_data(df: pd.DataFrame) -> bool:
    """
    Prepare sequence data with specific Feature Engineering:
    - Delta T
    - Energy calculation
    - Cumulative Energy (Window 50)
    - Log Transformation
    """
    global lstm_sequence_data, lstm_scaler
    
    if not (lstm_model and lstm_features):
        return False
    
    try:
        # Sort by time
        df_sort = df.sort_values('time').reset_index(drop=True)
        
        # Ensure numeric conversion (handling errors as per script)
        cols_to_numeric = ['magnitude', 'depth', 'latitude', 'longitude']
        for col in cols_to_numeric:
            df_sort[col] = pd.to_numeric(df_sort[col], errors='coerce')
        
        df_sort = df_sort.dropna(subset=cols_to_numeric)

        # --- Feature Engineering from Script ---
        
        # Calculate delta_t (time since last event in seconds)
        df_sort['delta_t'] = (df_sort['time'] - df_sort['time'].shift(1)).dt.total_seconds().fillna(0)
        
        # Energy Calculation: 10 ** (1.5 * mag + 4.8)
        df_sort['energy'] = 10 ** (1.5 * df_sort['magnitude'] + 4.8)
        
        # Cumulative energy released in the past N events (50 works best globally)
        window_energy = 50
        df_sort['cum_energy_50'] = df_sort['energy'].shift(1).rolling(window_energy, min_periods=1).sum()
        
        # Log it because the scale is huge
        df_sort['log_cum_energy_50'] = np.log10(df_sort['cum_energy_50'] + 1) # +1 to avoid log(0)
        
        # Fill first rows
        min_val = df_sort['log_cum_energy_50'].min()
        if pd.isna(min_val): min_val = 0
        df_sort['log_cum_energy_50'] = df_sort['log_cum_energy_50'].fillna(min_val)
        
        # Select features
        feature_cols = ['magnitude', 'depth', 'latitude', 'longitude', 'delta_t', 'log_cum_energy_50']
        data = df_sort[feature_cols].values
        
        if len(data) < LSTM_SEQ_LENGTH:
            print("Not enough data after processing for LSTM")
            return False

        # Fit Scaler (MinMaxScaler)
        lstm_scaler.fit(data)
        
        # Transform
        lstm_sequence_data = lstm_scaler.transform(data)
        
        # Save scaler for consistency
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(lstm_scaler, f)
            
        print(f"LSTM sequence ready: {len(lstm_sequence_data)} events processed with advanced energy features.")
        return True
    except Exception as e:
        print(f"LSTM prep failed: {e}")
        traceback.print_exc()
        return False

def train_lstm_model():
    """
    Train the LSTM model using the specific parameters provided:
    - Sequence Length: 40
    - Split: 80/20
    - EarlyStopping (patience=10)
    - Epochs: 50
    """
    global lstm_model
    
    if not KERAS_AVAILABLE or lstm_model is None or lstm_sequence_data is None:
        print("LSTM not available for training")
        return False
    
    if len(lstm_sequence_data) <= LSTM_SEQ_LENGTH:
        print("Not enough data for training")
        return False
    
    try:
        print("Preparing training sequences...")
        
        sequences = []
        labels = []
        
        # Create sequences
        for i in range(len(lstm_sequence_data) - LSTM_SEQ_LENGTH):
            sequences.append(lstm_sequence_data[i : i + LSTM_SEQ_LENGTH])
            # Predict next magnitude (index 0 is magnitude)
            labels.append(lstm_sequence_data[i + LSTM_SEQ_LENGTH, 0])
        
        X = np.array(sequences)
        y = np.array(labels)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training on {len(X_train)} sequences. Validation on {len(X_val)} sequences.")
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        print(f"Best validation loss: {min(history.history['val_loss'])}")
        
        # Save trained model
        lstm_model.save('earthquake_model.keras')
        print("LSTM model trained and saved")
        
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return False

def predict_next_earthquakes(req: LSTMPredictionRequest) -> List[Dict]:
    """Predict next n earthquakes using trained LSTM"""
    if not (lstm_model and lstm_sequence_data is not None and len(lstm_sequence_data) >= LSTM_SEQ_LENGTH):
        return []
    
    try:
        preds = []
        # Get the latest sequence from history
        seq_scaled = lstm_sequence_data[-LSTM_SEQ_LENGTH:].copy()
        feature_count = seq_scaled.shape[1] # Should be 6
        
        # Check if user provided manual inputs to override the very last step
        # This allows "What if" scenarios
        if all(v is not None for v in [req.magnitude, req.depth, req.latitude, req.longitude, req.delta_t, req.log_cum_energy_50]):
            user_features = np.array([[req.magnitude, req.depth, req.latitude, req.longitude, req.delta_t, req.log_cum_energy_50]])
            user_scaled = lstm_scaler.transform(user_features)
            seq_scaled[-1] = user_scaled[0]
            current_mag = req.magnitude
        else:
            # Unscale the last magnitude to have a reference
            dummy_last = np.zeros((1, feature_count))
            dummy_last[0] = seq_scaled[-1]
            current_mag = lstm_scaler.inverse_transform(dummy_last)[0, 0]

        for i in range(req.num_predictions):
            # Reshape for LSTM input (1, 40, 6)
            input_seq = seq_scaled.reshape(1, LSTM_SEQ_LENGTH, feature_count)
            
            # Predict scaled magnitude
            predicted_mag_scaled = lstm_model.predict(input_seq, verbose=0)[0][0]
            
            # Inverse transform
            # Construct a dummy row to use inverse_transform (we only care about index 0)
            dummy_row = np.zeros((1, feature_count))
            dummy_row[0, 0] = predicted_mag_scaled
            predicted_mag = lstm_scaler.inverse_transform(dummy_row)[0, 0]
            
            # --- Logic for Iterative Prediction ---
            # To predict step N+2, we need to generate features for step N+1
            
            # 1. Unscale previous step (last in sequence) to get baseline for location/depth
            prev_features_unscaled = lstm_scaler.inverse_transform(seq_scaled[-1].reshape(1, -1))[0]
            # [mag, depth, lat, lon, delta_t, log_energy]
            
            # 2. Heuristic updates for next features
            next_depth = prev_features_unscaled[1] # assume similar depth
            next_lat = prev_features_unscaled[2]   # assume same region
            next_lon = prev_features_unscaled[3]
            
            # Randomize delta_t slightly for realism (or use avg)
            next_delta_t = np.mean(lstm_scaler.inverse_transform(lstm_sequence_data)[-10:, 4])
            
            # Update Energy Features
            # New Energy
            new_energy = 10 ** (1.5 * predicted_mag + 4.8)
            
            # Previous Cumulative Energy (reverse log10)
            prev_log_cum = prev_features_unscaled[5]
            prev_cum = (10 ** prev_log_cum) - 1
            
            # New Cumulative
            new_cum = prev_cum + new_energy
            next_log_cum_energy = np.log10(new_cum + 1)
            
            # Assemble next vector
            next_features = np.array([[predicted_mag, next_depth, next_lat, next_lon, next_delta_t, next_log_cum_energy]])
            next_features_scaled = lstm_scaler.transform(next_features)
            
            # Update sequence: remove first, append new
            seq_scaled = np.vstack([seq_scaled[1:], next_features_scaled[0]])
            
            # Determine Risk
            if predicted_mag >= 6.0: risk = "High"
            elif predicted_mag >= 4.5: risk = "Medium"
            else: risk = "Low"
            
            preds.append({
                "prediction_number": i + 1,
                "predicted_magnitude": round(float(predicted_mag), 2),
                "risk": risk,
                "confidence": "Model Estimate"
            })
            
        return preds
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return []

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
        self.index = None
        self.metadata = []
        self.model = None
    
    def build(self, df: pd.DataFrame):
        """Build FAISS index from earthquake data"""
        if not VECTOR_SEARCH_AVAILABLE:
            return

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            texts = []
            self.metadata = []
            
            # Limit for vector search to avoid memory issues on free tiers
            sample_df = df.head(2000)
            
            for _, r in sample_df.iterrows():
                text = f"Magnitude {r['magnitude']} earthquake {r['place']} at depth {r['depth']}km on {r['timestamp_str']}"
                texts.append(text)
                
                self.metadata.append({
                    "id": str(r.get('id', '')),
                    "magnitude": float(r['magnitude']) if pd.notna(r['magnitude']) else 0.0,
                    "place": str(r['place']),
                    "timestamp": str(r['timestamp_str']),
                    "latitude": float(r['latitude']) if pd.notna(r['latitude']) else 0.0,
                    "longitude": float(r['longitude']) if pd.notna(r['longitude']) else 0.0,
                    "depth": float(r['depth']) if pd.notna(r['depth']) else 0.0,
                    "title": str(r.get('title', ''))
                })
            
            # Create embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Build FAISS index
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings.astype('float32'))
            
            print(f"Vector store built with {len(texts)} entries")
        except Exception as e:
            print(f"Vector store build failed: {e}")
    
    def search(self, query: str, k: int = 5):
        """Search for similar earthquake events"""
        if not self.index or not self.model:
            return []
        
        try:
            q_vec = self.model.encode([query])
            D, I = self.index.search(q_vec.astype('float32'), k)
            
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < len(self.metadata):
                    meta = self.metadata[idx].copy()
                    meta['similarity_score'] = float(1 / (1 + dist))
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
        self.docs = []
        self.meta = []
    
    def build(self, df: pd.DataFrame):
        """Build RAG index from earthquake data"""
        if not VECTOR_SEARCH_AVAILABLE:
            return

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            sample = df.head(500)
            
            for _, r in sample.iterrows():
                text = f"Magnitude {r['magnitude']} at {r['place']}, depth {r['depth']}km, time {r['timestamp_str']}"
                self.docs.append(text)
                
                self.meta.append({
                    "magnitude": float(r['magnitude']),
                    "place": str(r['place']),
                    "timestamp": str(r['timestamp_str']),
                    "depth": float(r['depth'])
                })
            
            # Create embeddings
            embs = self.model.encode(self.docs, convert_to_numpy=True, show_progress_bar=False)
            
            # Build FAISS index
            d = embs.shape[1]
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embs)
            self.index.add(embs.astype('float32'))
            
            print("RAG index ready")
        except Exception as e:
            print(f"RAG build failed: {e}")
    
    def query(self, question: str, k: int = 5):
        """Query RAG system with a question"""
        if not self.index or not self.model:
            return "RAG system not initialized", []
        
        try:
            # Encode query
            q = self.model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(q)
            
            # Search
            scores, idxs = self.index.search(q.astype('float32'), k)
            sources = [self.meta[i] for i in idxs[0] if i < len(self.meta)]
            
            # Build context for LLM
            context = "\n".join([
                f"{i+1}. M{s['magnitude']} {s['place']} ({s['timestamp']})" 
                for i, s in enumerate(sources[:3])
            ])
            
            prompt = f"Based on earthquake data, answer this question: {question}\n\nRelevant events:\n{context}\n\nProvide a concise answer:"
            
            # Try to call Groq API
            try:
                if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
                    payload = {
                        "model": GROQ_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 512
                    }
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}", 
                        "Content-Type": "application/json"
                    }
                    
                    r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=20)
                    
                    if r.status_code == 200:
                        answer = r.json()['choices'][0]['message']['content'].strip()
                    else:
                        answer = self._generate_simple_answer(question, sources)
                else:
                    answer = self._generate_simple_answer(question, sources)
            except Exception as e:
                print(f"LLM call failed: {e}")
                answer = self._generate_simple_answer(question, sources)
            
            return answer, sources
        except Exception as e:
            print(f"RAG query failed: {e}")
            return "Failed to process query", []
    
    def _generate_simple_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate a simple answer without LLM"""
        if not sources:
            return "No relevant earthquake data found for your question."
        
        # Extract statistics
        magnitudes = [s['magnitude'] for s in sources]
        places = [s['place'] for s in sources]
        
        avg_mag = sum(magnitudes) / len(magnitudes)
        max_mag = max(magnitudes)
        
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

# ------------------- Initialization -------------------
def init_system():
    """Initialize the entire system"""
    global earthquake_df
    
    print("Initializing system...")
    
    # Initialize LSTM models
    initialize_lstm_model()
    
    # Fetch earthquake data
    raw = fetch_earthquakes()
    df = process_data(raw)
    
    if df is None or len(df) == 0:
        print("Failed to load earthquake data")
        return False
    
    earthquake_df = df
    print(f"Loaded {len(df)} earthquakes")
    
    # Build vector store
    vector_store.build(df)
    
    # Build RAG system
    simple_rag.build(df)
    
    # Prepare LSTM data (Sequence generation + Feature Engineering)
    if lstm_model:
        prepare_lstm_sequence_data(df)
    
    # Train LSTM if not already trained
    if lstm_model and not os.path.exists('earthquake_model.keras'):
        train_lstm_model()
    
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
        "lstm_ready": lstm_model is not None,
        "lstm_data_ready": lstm_sequence_data is not None,
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
async def search(req: SearchRequest):
    """Semantic search for earthquakes"""
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
async def rag(req: RAGQueryRequest):
    """RAG-based question answering"""
    # Extract entities if available
    entities = []
    if SPACY_AVAILABLE and nlp_model:
        try:
            doc = nlp_model(req.question)
            entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
        except:
            pass
    
    # Query RAG system
    answer, sources = simple_rag.query(req.question)
    
    return {
        "question": req.question,
        "answer": answer,
        "extracted_entities": entities,
        "sources": sources[:3]
    }

@app.post("/api/predict/lstm")
async def lstm_forecast(req: LSTMPredictionRequest):
    """Forecast future earthquakes using LSTM with energy features"""
    if not lstm_model:
        raise HTTPException(
            status_code=503, 
            detail="LSTM model not available. Model files may be missing or incompatible."
        )
    
    if lstm_sequence_data is None or len(lstm_sequence_data) < LSTM_SEQ_LENGTH:
        raise HTTPException(
            status_code=503,
            detail="LSTM sequence data not prepared. Need more earthquake data for time series forecasting."
        )
    
    try:
        preds = predict_next_earthquakes(req)
        
        if not preds:
            raise HTTPException(status_code=500, detail="Forecast generation failed")
        
        # Calculate summary statistics
        magnitudes = [p['predicted_magnitude'] for p in preds]
        max_mag = max(magnitudes)
        avg_mag = sum(magnitudes) / len(magnitudes)
        min_mag = min(magnitudes)
        
        # Risk assessment
        if max_mag >= 6.0:
            risk = "High"
        elif max_mag >= 4.5:
            risk = "Moderate"
        else:
            risk = "Low"
        
        return {
            "predictions": preds,
            "summary": {
                "average_magnitude": round(avg_mag, 2),
                "maximum_magnitude": round(max_mag, 2),
                "minimum_magnitude": round(min_mag, 2),
                "risk_assessment": risk,
                "total_predictions": len(preds)
            },
            "disclaimer": "These predictions are based on an LSTM model using historical data patterns. Actual seismic activity is unpredictable.",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"LSTM forecast error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

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
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "path": str(request.url)
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "details": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting USGS Earthquake Intelligence System API...")
    print("Documentation available at: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)