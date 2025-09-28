# main.py - FINAL VERSION

import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List

from models import Node, Edge, VerificationPayload, RhythmPayload # Make sure all models are imported
from graph_manager import graph_db

app = FastAPI(
    title="BoB Kavach - Final Prototype API",
    version="3.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === In-memory Databases for the Prototype ===
challenges_db = {}
# This will store the "normal" typing rhythm for users. In a real app, use a proper database.
user_rhythm_profiles = {} 

# --- Lists for VKYC Challenge ---
EMOTIONS = ["Happy", "Surprised"]
PHRASES = [
    "Bank of Baroda provides secure banking",
    "Digital safety is our top priority",
    "My identity is secure with this technology"
]

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the BoB Kavach API. Go to /docs for documentation."}

# === VKYC Endpoints ===
@app.get("/vkyc/challenge", tags=["VKYC"])
def get_vkyc_challenge():
    challenge_id = str(random.randint(10000, 99999))
    challenge = {"emotion": random.choice(EMOTIONS), "phrase": random.choice(PHRASES)}
    challenges_db[challenge_id] = challenge
    return {"challenge_id": challenge_id, "challenge": challenge}

@app.post("/vkyc/verify", tags=["VKYC"])
def verify_vkyc_challenge(payload: Dict[str, Any]):
    challenge_id = payload.get("challenge_id")
    if not challenge_id or challenge_id not in challenges_db:
        raise HTTPException(status_code=400, detail="Invalid or expired challenge ID.")
    expected = challenges_db.pop(challenge_id)
    
    spoken_phrase = payload.get("spoken_phrase", "").lower().strip()
    expected_phrase = expected["phrase"].lower().strip()
    spoken_words = set(spoken_phrase.split())
    expected_words = set(expected_phrase.split())
    common_words = spoken_words.intersection(expected_words)
    phrase_correct = (len(common_words) / len(expected_words)) > 0.7

    blendshapes = payload.get("blendshapes", {})
    emotion_correct = False
    if expected["emotion"] == "Happy" and blendshapes.get("mouthSmileLeft", 0) > 0.4 and blendshapes.get("mouthSmileRight", 0) > 0.4:
        emotion_correct = True
    elif expected["emotion"] == "Surprised" and blendshapes.get("eyeWideLeft", 0) > 0.3 and blendshapes.get("jawOpen", 0) > 0.2:
        emotion_correct = True
            
    if phrase_correct and emotion_correct:
        return {"status": "Success", "message": "Verification successful!"}
    else:
        return {"status": "Failed", "message": "Liveness check failed.", "details": {"phrase_matched": phrase_correct, "emotion_matched": emotion_correct}}

# === NEW: Session Rhythm Analysis Endpoint ===
@app.post("/analyze-rhythm", tags=["Session Analysis"])
def analyze_typing_rhythm(payload: RhythmPayload):
    user_id = payload.user_id
    timings = payload.timings
    
    if not timings:
        raise HTTPException(status_code=400, detail="No timing data provided.")
    
    current_avg_timing = np.mean(timings)

    # If we have no profile for this user, we create one. This is their "normal" rhythm.
    if user_id not in user_rhythm_profiles:
        user_rhythm_profiles[user_id] = {
            "mean": current_avg_timing,
            "std_dev": np.std(timings) if len(timings) > 1 else 30  # Default std dev
        }
        return {"status": "Profile Created", "message": "User's normal typing rhythm has been saved."}
    
    # If a profile exists, we compare the current rhythm to their normal one.
    profile = user_rhythm_profiles[user_id]
    mean = profile["mean"]
    std_dev = profile["std_dev"]

    # Calculate Z-score: how many standard deviations away is this attempt?
    z_score = abs((current_avg_timing - mean) / (std_dev or 1))
    
    if z_score < 2.0: # If it's within 2 standard deviations, it's considered normal.
        return {"status": "Rhythm Matched", "z_score": z_score}
    else:
        return {"status": "Rhythm Mismatch", "z_score": z_score, "message": "Warning: Typing rhythm is abnormal."}

# === Fraud Ring Endpoints (Unchanged) ===
@app.post("/nodes", tags=["Graph Management"])
def create_node(node: Node):
    result = graph_db.add_node(node.node_id, node.node_type)
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return result

@app.post("/edges", tags=["Graph Management"])
def create_edge(edge: Edge):
    result = graph_db.add_edge(edge.source_id, edge.target_id)
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return result

@app.put("/nodes/{node_id}/flag-fraud", tags=["Fraud Management"])
def flag_node(node_id: str):
    result = graph_db.flag_node_as_fraud(node_id)
    if result["status"] == "error": raise HTTPException(404, result["message"])
    return result

@app.get("/risk-check/{user_id}", tags=["Fraud Management"])
def check_risk(user_id: str, max_depth: Optional[int] = 3):
    return graph_db.check_fraud_risk(user_id, max_depth)
