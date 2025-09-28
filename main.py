# main.py

import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

# Import all models
from models import Node, Edge, VerificationPayload

# Import the graph manager
from graph_manager import graph_db

app = FastAPI(
    title="BoB Kavach - Upgraded Prototype API",
    version="2.0.0"
)

# --- CORS Middleware (Crucial for frontend-backend communication) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory store for our new challenges ---
challenges_db = {}
# Lists of emotions and phrases for the new challenge
EMOTIONS = ["Happy", "Surprised"]
PHRASES = [
    "Bank of Baroda provides secure banking",
    "Digital safety is our top priority",
    "My identity is secure with this technology",
    "I am completing my video verification"
]

# === Upgraded VKYC Endpoints ===

@app.get("/vkyc/challenge", tags=["VKYC"])
def get_vkyc_challenge():
    """
    Generates a new, more advanced Emotion & Text challenge.
    """
    challenge_id = str(random.randint(10000, 99999))
    challenge = {
        "emotion": random.choice(EMOTIONS),
        "phrase": random.choice(PHRASES)
    }
    challenges_db[challenge_id] = challenge
    return {"challenge_id": challenge_id, "challenge": challenge}

@app.post("/vkyc/verify", tags=["VKYC"])
def verify_vkyc_challenge(payload: Dict[str, Any]):
    """
    Verifies the user's response to the Emotion & Text challenge.
    We are using a flexible Dict for the payload to accept blendshape data.
    """
    challenge_id = payload.get("challenge_id")
    if not challenge_id or challenge_id not in challenges_db:
        raise HTTPException(status_code=400, detail="Invalid or expired challenge ID.")

    expected = challenges_db.pop(challenge_id)
    
    # 1. Verify the spoken phrase (using a similarity check)
    spoken_phrase = payload.get("spoken_phrase", "").lower().strip()
    expected_phrase = expected["phrase"].lower().strip()
    # A simple check for this prototype: check if more than 70% of words match
    spoken_words = set(spoken_phrase.split())
    expected_words = set(expected_phrase.split())
    common_words = spoken_words.intersection(expected_words)
    phrase_correct = (len(common_words) / len(expected_words)) > 0.7

    # 2. Verify the facial emotion using blendshapes from MediaPipe
    blendshapes = payload.get("blendshapes", {})
    emotion_correct = False
    
    # This is a simplified, rule-based emotion classifier perfect for a prototype
    if expected["emotion"] == "Happy":
        # A "Happy" face usually involves smiling
        if blendshapes.get("mouthSmileLeft", 0) > 0.4 and blendshapes.get("mouthSmileRight", 0) > 0.4:
            emotion_correct = True
    elif expected["emotion"] == "Surprised":
        # A "Surprised" face usually involves open eyes and a slightly open mouth
        if blendshapes.get("eyeWideLeft", 0) > 0.3 and blendshapes.get("eyeWideRight", 0) > 0.3 and blendshapes.get("jawOpen", 0) > 0.2:
            emotion_correct = True
            
    # 3. Final Decision
    if phrase_correct and emotion_correct:
        return {"status": "Success", "message": "Verification successful!"}
    else:
        return {
            "status": "Failed",
            "message": "Liveness check failed. Please try again.",
            "details": {
                "phrase_matched": phrase_correct,
                "emotion_matched": emotion_correct,
            }
        }

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