import os
import asyncio
import json
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
    DeepgramClientOptions
)

from depression_detector import DepressionDetector

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(title="Speech Biomarker Depression Detection")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the depression detector
depression_detector = DepressionDetector()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Initialize Deepgram client
try:
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")
    
    deepgram = DeepgramClient(deepgram_api_key)
    logger.info("Deepgram client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Deepgram client: {e}")
    deepgram = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Get API key from query parameters
        query_params = dict(websocket.query_params)
        user_api_key = query_params.get("api_key", "")
        
        if not user_api_key:
            await manager.send_message(json.dumps({"error": "No Deepgram API key provided. Please provide your API key."}), websocket)
            return
        
        # Initialize Deepgram client with user's API key
        try:
            user_deepgram = DeepgramClient(user_api_key, DeepgramClientOptions(options={}))
            logger.info("Deepgram client initialized with user's API key")
        except Exception as e:
            error_msg = f"Failed to initialize Deepgram client with provided API key: {e}"
            logger.error(error_msg)
            await manager.send_message(json.dumps({"error": error_msg}), websocket)
            return
        
        # Set up Deepgram live transcription
        options = LiveOptions(
            model="nova-3",
            language="en",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            smart_format=True,
            interim_results=True,
            utterance_end_ms=1000,
            vad_events=True
        )
        
        # Create live transcription connection using user's API key
        dg_connection = user_deepgram.listen.live.v("1")
        
        # Define event handlers
        @dg_connection.on(LiveTranscriptionEvents.Transcript)
        async def handle_transcript(transcript):
            transcript_dict = transcript.to_dict()
            
            # Check if we have a transcript to process
            if transcript_dict.get("is_final") and "channel" in transcript_dict:
                if transcript_dict["channel"]["alternatives"]:
                    spoken_text = transcript_dict["channel"]["alternatives"][0].get("transcript", "")
                    
                    if spoken_text.strip():
                        # Process text for depression biomarkers
                        depression_score, features = depression_detector.analyze_text(spoken_text)
                        
                        # Create response with transcript and depression analysis
                        response = {
                            "transcript": spoken_text,
                            "depression_score": depression_score,
                            "depression_level": depression_detector.get_depression_level(depression_score),
                            "features": features
                        }
                        
                        # Send results back to the client
                        await manager.send_message(json.dumps(response), websocket)
                        
                        # Log the analysis (optional)
                        logger.info(f"Depression analysis: {depression_score} - '{spoken_text[:50]}...'")
        
        # Listen for WebSocket messages (audio data)
        while True:
            audio_data = await websocket.receive_bytes()
            
            # Send audio data to Deepgram
            if dg_connection.is_ready():
                dg_connection.send(audio_data)
            else:
                # If the connection is not ready yet, inform the client
                await manager.send_message(json.dumps({"error": "Deepgram connection not ready"}), websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if "dg_connection" in locals():
            dg_connection.finish()
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)
        if "dg_connection" in locals():
            dg_connection.finish()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
