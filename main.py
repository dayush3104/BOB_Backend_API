# main.py - FINAL GEMINI-POWERED VERSION

import os
import re
import json
import random
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from models import Node, Edge, RhythmPayload
from graph_manager import graph_db

# --- Configure the Gemini API ---
# The code reads the API key from an environment variable named "GOOGLE_API_KEY"
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    # This will cause a clean error on startup if the key is missing on Render
    print("WARNING: Google API Key not found in environment. API calls will fail.")
    # In a real app you might want to raise ValueError("Google API Key not found...")

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')


app = FastAPI(
    title="BoB Kavach - Gemini-Powered API",
    version="4.0.0"
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
user_rhythm_profiles = {} 
EMOTIONS = ["Happy", "Surprised"]
PHRASES = [
    "Bank of Baroda provides secure banking",
    "Digital safety is our top priority",
    "My identity is secure with this technology"
]

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Gemini-Powered BoB Kavach API. Go to /docs for documentation."}

# === VKYC Endpoints (Now Gemini-Powered) ===
@app.get("/vkyc/challenge", tags=["VKYC"])
def get_vkyc_challenge():
    challenge_id = str(random.randint(10000, 99999))
    challenge = {"emotion": random.choice(EMOTIONS), "phrase": random.choice(PHRASES)}
    challenges_db[challenge_id] = challenge
    return {"challenge_id": challenge_id, "challenge": challenge}

@app.post("/vkyc/verify", tags=["VKYC"])
async def verify_vkyc_with_gemini(
    challenge_id: str = Form(...),
    expected_emotion: str = Form(...),
    expected_phrase: str = Form(...),
    video_file: UploadFile = File(...)
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server is missing API Key configuration.")
        
    if challenge_id not in challenges_db or challenges_db.pop(challenge_id) is None:
        raise HTTPException(status_code=400, detail="Invalid or expired challenge ID.")

    # 1. Upload the video file for Gemini to process
    print("Uploading video file to Gemini...")
    video_file_for_api = genai.upload_file(
        path=video_file.file,
        display_name=video_file.filename
    )
    print(f"File upload complete. URI: {video_file_for_api.uri}")

    # 2. Create the prompt for the Gemini API
    prompt = f"""
    You are a liveness detection expert for a bank's Video KYC process.
    Analyze this video. The user was instructed to show the emotion '{expected_emotion}' and say the phrase '{expected_phrase}'.
    
    Carefully analyze the video and determine two things:
    1. Did the user's spoken words closely match the required phrase?
    2. Did the user's facial expression genuinely match the requested emotion?

    Respond ONLY with a valid JSON object in the format:
    {{"phrase_matched": boolean, "emotion_matched": boolean, "reasoning": "A brief explanation of your decision."}}
    """

    # 3. Call the Gemini API
    print("Sending request to Gemini API...")
    try:
        response = gemini_model.generate_content([prompt, video_file_for_api])
        
        # Clean up the response to extract the JSON
        text_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(text_response)
        
        phrase_matched = result.get("phrase_matched", False)
        emotion_matched = result.get("emotion_matched", False)

        if phrase_matched and emotion_matched:
            return {"status": "Success", "message": "Verification successful!", "details": result.get("reasoning")}
        else:
            return {"status": "Failed", "message": "Liveness check failed.", "details": result.get("reasoning")}

    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process video with AI model. Error: {str(e)}")


# === Session Rhythm Analysis & Fraud Ring Endpoints (Unchanged) ===
# ... (the rest of your main.py file from the previous version) ...
@app.post("/analyze-rhythm", tags=["Session Analysis"])
def analyze_typing_rhythm(payload: RhythmPayload):
    user_id = payload.user_id
    timings = payload.timings
    if not timings:
        raise HTTPException(status_code=400, detail="No timing data provided.")
    current_avg_timing = np.mean(timings)
    if user_id not in user_rhythm_profiles:
        user_rhythm_profiles[user_id] = {"mean": current_avg_timing, "std_dev": np.std(timings) if len(timings) > 1 else 30}
        return {"status": "Profile Created", "message": "User's normal typing rhythm has been saved."}
    profile = user_rhythm_profiles[user_id]
    z_score = abs((current_avg_timing - profile["mean"]) / (profile["std_dev"] or 1))
    if z_score < 2.0:
        return {"status": "Rhythm Matched", "z_score": z_score}
    else:
        return {"status": "Rhythm Mismatch", "z_score": z_score, "message": "Warning: Typing rhythm is abnormal."}

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
