"""
RobustNLP Enterprise Integration API Server
This FastAPI server provides the API endpoints for the Enterprise Integration Web Dashboard
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Import our modules
from threat_engine import ThreatEngine
from defense_engine import DefenseEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RobustNLP Enterprise Integration API",
    description="API endpoints for the RobustNLP Enterprise Integration Showcase",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our engines
threat_engine = ThreatEngine()
defense_engine = DefenseEngine()

# Session storage for chatbot conversations
chatbot_sessions = {}

# Pydantic models for request validation
class ToxicRequest(BaseModel):
    text: str
    apply_defense: bool = True
    defense_level: str = "standard"  # minimal, standard, aggressive

class ChatbotRequest(BaseModel):
    message: str
    defense_enabled: bool = True

class SpamRequest(BaseModel):
    content: str
    subject: Optional[str] = None
    apply_defense: bool = True

class LLMGuardianRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5"

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Example usage in the toxic comment analysis endpoint
@app.post("/toxic")
async def analyze_toxic_comment(request: ToxicRequest):
    try:
        logger.info(f"Analyzing toxic comment: {request.text[:50]}...")
        logger.info(f"ThreatEngine object: {threat_engine}")
        logger.info(f"ThreatEngine analyze_text method: {hasattr(threat_engine, 'analyze_text')}")
        # Analyze the text for adversarial threats
        threat_result = threat_engine.analyze_text(
            text=request.text,
            task_type="toxicity_detection"
        )
        
        # Apply defenses if requested
        defended_text = request.text
        defense_applied = False
        
        if request.apply_defense and (threat_result["threat_detected"] or threat_result["toxicity_score"] > 0.7):
            defense_result = defense_engine.defend_text(
                text=request.text,
                task_type="toxicity_detection",
                threat_info=threat_result,
                defense_level=request.defense_level
            )
            defended_text = defense_result["defended_text"]
            defense_applied = True
            
            # Re-analyze after defense
            threat_result = threat_engine.analyze_text(
                text=defended_text,
                task_type="toxicity_detection"
            )
        
        # Prepare response
        return {
            "original_text": request.text,
            "defended_text": defended_text if defense_applied else None,
            "is_toxic": threat_result["toxicity_score"] > 0.5,
            "toxicity_score": threat_result["toxicity_score"],
            "threat_detected": threat_result["threat_detected"],
            "threat_type": threat_result.get("threat_type", None) if threat_result["threat_detected"] else None,
            "threat_score": threat_result.get("threat_score", 0.0) if threat_result["threat_detected"] else 0.0,
            "defense_applied": defense_applied
        }
    except Exception as e:
        logger.error(f"Error in toxic comment analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing comment: {str(e)}")
# Chatbot endpoint
@app.post("/chatbot")
async def chatbot_message(request: ChatbotRequest, client_ip: str = None):
    try:
        # Use client IP as session ID for demo purposes
        # In production, use proper session management
        session_id = client_ip or "default_session"
        
        if session_id not in chatbot_sessions:
            chatbot_sessions[session_id] = {
                "history": [],
                "system_prompt": "You are a helpful assistant for a business. Answer questions professionally and concisely."
            }
        
        # Add message to history
        chatbot_sessions[session_id]["history"].append({
            "role": "user",
            "content": request.message
        })
        
        # Check for threats
        threat_result = threat_engine.analyze_text(
            text=request.message,
            task_type="chatbot_prompt",
            conversation_history=chatbot_sessions[session_id]["history"]
        )
        
        message_to_process = request.message
        defense_applied = False
        
        # Apply defense if threat detected and defense is enabled
        if request.defense_enabled and threat_result["threat_detected"]:
            defense_result = defense_engine.defend_text(
                text=request.message,
                task_type="chatbot_prompt",
                threat_info=threat_result,
                conversation_history=chatbot_sessions[session_id]["history"]
            )
            message_to_process = defense_result["defended_text"]
            defense_applied = True
        
        # Generate response (in a real system, this would call an LLM)
        # Here we simulate a response
        if threat_result["threat_detected"] and not defense_applied:
            response = "I'm sorry, but I cannot respond to that request as it appears to violate our usage policies."
        else:
            response = "I understand your message. How else can I help you today?"
        
        # Add response to history
        chatbot_sessions[session_id]["history"].append({
            "role": "assistant",
            "content": response
        })
        
        return {
            "response": response,
            "threat_detected": threat_result["threat_detected"],
            "threat_type": threat_result.get("threat_type", None) if threat_result["threat_detected"] else None,
            "defense_applied": defense_applied
        }
        
    except Exception as e:
        logger.error(f"Error in chatbot message processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))