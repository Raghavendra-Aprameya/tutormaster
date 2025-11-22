"""
FastAPI WebSocket server for Interactive Teaching Agent
Exposes the teaching agent via WebSocket for real-time communication
Includes ElevenLabs TTS for voice responses
"""

import os
import json
import asyncio
import base64
import sys
import tempfile
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Body, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import aiohttp
from dotenv import load_dotenv

from teaching_agent import TeachingAgent
from exam_question_generator import ExamQuestionGenerator, SUBJECT, CLASS_LEVEL, CHAPTER
from answer_evaluator import AnswerEvaluator
from revision_pointers_generator import RevisionPointersGenerator

# Import embedding functions from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_embeddings import ingest_file_to_chroma

# Load environment variables
load_dotenv()


# Initialize FastAPI app
app = FastAPI(
    title="Interactive Teaching Agent API",
    description="WebSocket-based AI teaching assistant for History Class 10",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections and teaching agent instances"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.teaching_agents: Dict[str, TeachingAgent] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # Create new teaching agent for this connection
        self.teaching_agents[client_id] = TeachingAgent()
        print(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """Remove connection and clean up"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.teaching_agents:
            del self.teaching_agents[client_id]
        print(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
    
    async def send_message(self, client_id: str, message: dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending message to {client_id}: {str(e)}")
                # Connection might be closed, remove it
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                if client_id in self.teaching_agents:
                    del self.teaching_agents[client_id]
    
    def get_agent(self, client_id: str) -> TeachingAgent:
        """Get teaching agent for specific client"""
        return self.teaching_agents.get(client_id)


# Create connection manager instance
manager = ConnectionManager()


# ElevenLabs TTS Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice: Rachel
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"


async def generate_tts_audio(text: str, voice_id: str = ELEVENLABS_VOICE_ID) -> Optional[bytes]:
    """
    Generate TTS audio using ElevenLabs API
    
    Args:
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID (default from env)
        
    Returns:
        Audio bytes (MP3 format) or None if error
    """
    if not ELEVENLABS_API_KEY:
        print("Warning: ELEVENLABS_API_KEY not set. TTS will be disabled.")
        return None
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{ELEVENLABS_API_URL}/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    return audio_bytes
                else:
                    error_text = await response.text()
                    print(f"ElevenLabs API error: {response.status} - {error_text}")
                    return None
    except Exception as e:
        print(f"Error generating TTS audio: {str(e)}")
        return None


async def stream_response_with_tts(websocket: WebSocket, text: str, metadata: dict):
    """
    Stream text response and generate TTS audio
    
    Args:
        websocket: WebSocket connection
        text: Response text to stream
        metadata: Additional metadata to include
    """
    try:
        # Send text response immediately
        await websocket.send_json({
            "type": "response",
            "response": text,
            **metadata,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"Error sending text response: {str(e)}")
        raise
    
    try:
        # Generate and send audio
        audio_bytes = await generate_tts_audio(text)
        if audio_bytes:
            # Convert audio to base64 for JSON transmission
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            await websocket.send_json({
                "type": "audio",
                "audio": audio_base64,
                "format": "mp3",
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            # Send error if TTS fails (only if connection is still open)
            try:
                await websocket.send_json({
                    "type": "audio_error",
                    "message": "Failed to generate audio",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception:
                # Connection might be closed, ignore
                pass
    except Exception as e:
        print(f"Error sending audio: {str(e)}")
        # Don't raise - audio failure shouldn't break the connection


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Interactive Teaching Agent WebSocket API",
        "version": "1.0.0",
        "websocket_endpoint": "/ws/{client_id}",
        "status": "active",
        "subject": "History",
        "class_level": "10",
        "chapter": "A Brief History of India"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/exam-questions")
async def get_exam_questions():
    """
    Generate 10 exam-style questions and answers from the document.
    
    Returns:
        JSON response with 10 exam questions and answers
    """
    try:
        # Initialize generator
        generator = ExamQuestionGenerator()
        
        # Generate exactly 10 questions
        result = generator.generate_exam_questions(num_questions=10)
        
        return {
            "success": True,
            "subject": SUBJECT,
            "class_level": CLASS_LEVEL,
            "chapter": CHAPTER,
            "total_questions": len(result),
            "questions": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error generating exam questions: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/api/exam-questions")
async def generate_exam_questions():
    """
    Generate 10 exam-style questions and answers from the document (POST endpoint).
    
    Returns:
        JSON response with 10 exam questions and answers
    """
    try:
        # Initialize generator
        generator = ExamQuestionGenerator()
        
        # Generate exactly 10 questions
        result = generator.generate_exam_questions(num_questions=10)
        
        return {
            "success": True,
            "subject": SUBJECT,
            "class_level": CLASS_LEVEL,
            "chapter": CHAPTER,
            "total_questions": len(result),
            "questions": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error generating exam questions: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/api/evaluate-answer")
async def evaluate_answer(request: Request):
    """
    Evaluate a student's answer against the correct answer.
    
    Request body:
    {
        "question": "Question text",
        "answer": "Correct answer",
        "student_answer": "Student's answer"
    }
    
    Returns:
        JSON response with score (out of 10), feedback, and evaluation details
    """
    try:
        # Parse request body
        body = await request.json()
        
        # Validate request
        if not body:
            return {
                "success": False,
                "error": "Request body is required",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        question = body.get("question", "").strip()
        correct_answer = body.get("answer", "").strip()
        student_answer = body.get("student_answer", "").strip()
        
        if not question:
            return {
                "success": False,
                "error": "Question is required",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if not correct_answer:
            return {
                "success": False,
                "error": "Correct answer is required",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize evaluator
        evaluator = AnswerEvaluator()
        
        # Evaluate answer
        evaluation = evaluator.evaluate_answer(question, correct_answer, student_answer)
        
        return {
            "success": True,
            "question": question,
            "evaluation": evaluation,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error evaluating answer: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Request schema for revision pointers
class RevisionPointersRequest(BaseModel):
    study_material_id: Optional[str] = None
    subject: Optional[str] = None
    class_level: Optional[str] = None
    chapter: Optional[str] = None


@app.get("/api/revision-pointers")
async def get_revision_pointers(
    study_material_id: Optional[str] = Query(None, description="Study Material ID (UUID) from PostgreSQL"),
    subject: Optional[str] = Query(None, description="Subject name"),
    class_level: Optional[str] = Query(None, description="Class level"),
    chapter: Optional[str] = Query(None, description="Chapter name")
):
    """
    Generate last-minute revision pointers from the entire chapter (GET endpoint).
    
    Query parameters (optional):
    - study_material_id: UUID from PostgreSQL (primary filter)
    - subject: Subject name
    - class_level: Class level
    - chapter: Chapter name
    
    Returns:
        JSON response with a list of revision pointers
    """
    try:
        # Initialize generator with parameters
        generator = RevisionPointersGenerator(
            subject=subject,
            class_level=class_level,
            chapter=chapter,
            study_material_id=study_material_id
        )
        
        print(f"Generating revision pointers - study_material_id: {study_material_id}, subject: {subject}, class_level: {class_level}, chapter: {chapter}")
        
        # Generate revision pointers
        result = generator.generate_revision_pointers()
        
        return {
            "success": True,
            "subject": result["subject"],
            "class_level": result["class_level"],
            "chapter": result["chapter"],
            "pointers": result["pointers"],
            "total_pointers": result["total_pointers"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error generating revision pointers: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/api/revision-pointers")
async def generate_revision_pointers(request: RevisionPointersRequest = Body(None)):
    """
    Generate last-minute revision pointers from the entire chapter (POST endpoint).
    
    Request body (optional):
    {
        "study_material_id": "uuid-from-postgres",
        "subject": "History",
        "class_level": "10",
        "chapter": "Chapter 1 Notes"
    }
    
    If study_material_id is provided, it will filter embeddings by that ID from ChromaDB.
    If not provided, uses subject, class_level, and chapter to filter.
    
    Returns:
        JSON response with a list of revision pointers
    """
    try:
        # Parse request body if provided
        subject = None
        class_level = None
        chapter = None
        study_material_id = None
        
        if request:
            study_material_id = request.study_material_id
            subject = request.subject
            class_level = request.class_level
            chapter = request.chapter
        
        # Initialize generator with parameters
        generator = RevisionPointersGenerator(
            subject=subject,
            class_level=class_level,
            chapter=chapter,
            study_material_id=study_material_id
        )
        
        print(f"Generating revision pointers - study_material_id: {study_material_id}, subject: {subject}, class_level: {class_level}, chapter: {chapter}")
        
        # Generate revision pointers
        result = generator.generate_revision_pointers()
        
        return {
            "success": True,
            "subject": result["subject"],
            "class_level": result["class_level"],
            "chapter": result["chapter"],
            "pointers": result["pointers"],
            "total_pointers": result["total_pointers"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error generating revision pointers: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/api/evaluate-answers")
async def evaluate_multiple_answers(request: Request):
    """
    Evaluate multiple student answers at once.
    
    Request body:
    {
        "evaluations": [
            {
                "question": "Question 1",
                "answer": "Correct answer 1",
                "student_answer": "Student answer 1"
            },
            {
                "question": "Question 2",
                "answer": "Correct answer 2",
                "student_answer": "Student answer 2"
            }
        ]
    }
    
    Returns:
        JSON response with overall results and individual evaluations
    """
    try:
        # Parse request body
        body = await request.json()
        
        # Validate request
        if not body or "evaluations" not in body:
            return {
                "success": False,
                "error": "Request body must contain 'evaluations' array",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        evaluations = body.get("evaluations", [])
        
        if not isinstance(evaluations, list) or len(evaluations) == 0:
            return {
                "success": False,
                "error": "Evaluations must be a non-empty array",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize evaluator
        evaluator = AnswerEvaluator()
        
        # Evaluate all answers
        result = evaluator.evaluate_multiple_answers(evaluations)
        
        return {
            "success": True,
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error evaluating answers: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Webhook endpoint for creating embeddings
class CreateEmbeddingRequest(BaseModel):
    file_url: str
    document_id: str
    subject: str
    class_level: str
    title: str
    filename: Optional[str] = None


class RevisionPointersRequest(BaseModel):
    study_material_id: Optional[str] = None
    subject: Optional[str] = None
    class_level: Optional[str] = None
    chapter: Optional[str] = None


@app.post("/api/create-embeddings")
async def create_embeddings_webhook(request: CreateEmbeddingRequest = Body(...)):
    """
    Webhook endpoint to create embeddings for a study material document.
    
    This endpoint:
    1. Downloads the file from Cloudinary URL
    2. Processes it and creates vector embeddings
    3. Stores embeddings in ChromaDB
    
    Request body:
    {
        "file_url": "https://res.cloudinary.com/...",
        "document_id": "uuid-of-study-material",
        "subject": "History",
        "class_level": "10",
        "title": "Chapter 1 Notes",
        "filename": "document.pdf" (optional)
    }
    """
    temp_file_path = None
    try:
        # Download file from Cloudinary URL
        async with aiohttp.ClientSession() as session:
            async with session.get(request.file_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download file from URL: {response.status}"
                    )
                
                # Create temporary file
                file_extension = request.filename.split('.')[-1] if request.filename and '.' in request.filename else 'pdf'
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
                temp_file_path = temp_file.name
                
                # Write downloaded content to temp file
                content = await response.read()
                temp_file.write(content)
                temp_file.close()
        
        # Process file and create embeddings (run in executor to avoid blocking)
        # request.document_id is the study_material_id from PostgreSQL
        study_material_id = request.document_id
        loop = asyncio.get_event_loop()
        document_id = await loop.run_in_executor(
            None,
            ingest_file_to_chroma,
            temp_file_path,
            request.subject,
            request.title,  # chapter = title
            request.class_level,
            study_material_id,  # document_id = study_material_id
            study_material_id   # study_material_id for metadata
        )
        
        return {
            "success": True,
            "message": "Embeddings created successfully",
            "document_id": document_id,
            "chunks_ingested": "See logs for details",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating embeddings: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error deleting temp file: {str(e)}")


@app.get("/demo", response_class=HTMLResponse)
async def demo_client():
    """Simple HTML demo client for testing WebSocket"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Teaching Agent WebSocket Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }
            #chat-container {
                border: 1px solid #ccc;
                height: 400px;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e3f2fd;
                text-align: right;
            }
            .agent-message {
                background-color: #f1f8e9;
            }
            .system-message {
                background-color: #fff3e0;
                font-style: italic;
            }
            #input-container {
                display: flex;
                gap: 10px;
            }
            #message-input {
                flex: 1;
                padding: 10px;
                font-size: 14px;
            }
            #send-button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                font-size: 14px;
            }
            #send-button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            #status {
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            .connected { background-color: #c8e6c9; }
            .disconnected { background-color: #ffcdd2; }
            .audio-indicator {
                display: inline-block;
                margin-left: 10px;
                font-size: 12px;
                color: #666;
            }
            .audio-playing {
                color: #4CAF50;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>Interactive Teaching Agent - WebSocket Demo</h1>
        <div id="status" class="disconnected">Status: Disconnected</div>
        <div id="audio-status" class="audio-indicator"></div>
        
        <div id="chat-container"></div>
        
        <div id="input-container">
            <input type="text" id="message-input" placeholder="Type your message..." disabled>
            <button id="send-button" disabled>Send</button>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>Try these commands:</h3>
            <ul>
                <li>"teach me about ancient India"</li>
                <li>"tell me more"</li>
                <li>"what year was the Indus Valley discovered?"</li>
                <li>"continue"</li>
            </ul>
        </div>

        <script>
            const clientId = 'demo_' + Math.random().toString(36).substr(2, 9);
            let ws = null;
            
            const chatContainer = document.getElementById('chat-container');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const statusDiv = document.getElementById('status');
            const audioStatusDiv = document.getElementById('audio-status');
            
            function connect() {
                ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
                
                ws.onopen = function(event) {
                    statusDiv.textContent = 'Status: Connected';
                    statusDiv.className = 'connected';
                    messageInput.disabled = false;
                    sendButton.disabled = false;
                    addMessage('system', 'Connected to teaching agent!');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onerror = function(error) {
                    addMessage('system', 'WebSocket error occurred');
                    console.error('WebSocket error:', error);
                };
                
                ws.onclose = function(event) {
                    statusDiv.textContent = 'Status: Disconnected';
                    statusDiv.className = 'disconnected';
                    messageInput.disabled = true;
                    sendButton.disabled = true;
                    addMessage('system', 'Disconnected from server');
                };
            }
            
            let currentAudio = null;
            let audioQueue = [];
            let isPlayingAudio = false;
            
            function handleMessage(data) {
                if (data.type === 'response') {
                    addMessage('agent', data.response);
                    
                    // Add metadata if available
                    if (data.mode) {
                        addMessage('system', `Mode: ${data.mode}`);
                    }
                } else if (data.type === 'audio') {
                    // Handle audio data
                    handleAudio(data);
                } else if (data.type === 'audio_error') {
                    addMessage('system', `Audio Error: ${data.message}`);
                } else if (data.type === 'error') {
                    addMessage('system', `Error: ${data.message}`);
                }
            }
            
            function handleAudio(data) {
                // Decode base64 audio
                const audioData = atob(data.audio);
                const audioArray = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    audioArray[i] = audioData.charCodeAt(i);
                }
                
                // Create blob and audio element
                const blob = new Blob([audioArray], { type: 'audio/mpeg' });
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                
                // Add to queue
                audioQueue.push(audio);
                
                // Play if not already playing
                if (!isPlayingAudio) {
                    playNextAudio();
                }
            }
            
            function playNextAudio() {
                if (audioQueue.length === 0) {
                    isPlayingAudio = false;
                    audioStatusDiv.textContent = '';
                    audioStatusDiv.classList.remove('audio-playing');
                    return;
                }
                
                isPlayingAudio = true;
                audioStatusDiv.textContent = '🔊 Playing audio...';
                audioStatusDiv.classList.add('audio-playing');
                
                const audio = audioQueue.shift();
                
                audio.onended = function() {
                    URL.revokeObjectURL(audio.src);
                    playNextAudio();
                };
                
                audio.onerror = function(e) {
                    console.error('Audio playback error:', e);
                    URL.revokeObjectURL(audio.src);
                    audioStatusDiv.textContent = '❌ Audio error';
                    playNextAudio();
                };
                
                audio.play().catch(function(error) {
                    console.error('Error playing audio:', error);
                    URL.revokeObjectURL(audio.src);
                    audioStatusDiv.textContent = '❌ Playback failed';
                    playNextAudio();
                });
            }
            
            function addMessage(sender, text) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + sender + '-message';
                
                const timestamp = new Date().toLocaleTimeString();
                const senderLabel = sender === 'user' ? 'You' : 
                                   sender === 'agent' ? 'Teacher' : 'System';
                
                messageDiv.innerHTML = `<strong>${senderLabel}</strong> (${timestamp})<br>${text}`;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function sendMessage() {
                const message = messageInput.value.trim();
                if (message && ws && ws.readyState === WebSocket.OPEN) {
                    addMessage('user', message);
                    ws.send(JSON.stringify({
                        type: 'message',
                        content: message
                    }));
                    messageInput.value = '';
                }
            }
            
            sendButton.onclick = sendMessage;
            
            messageInput.onkeypress = function(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            };
            
            // Connect on load
            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time communication with teaching agent
    
    Message format (client -> server):
    {
        "type": "message",
        "content": "user message text"
    }
    
    Response format (server -> client):
    {
        "type": "response",
        "response": "agent response text",
        "mode": "teaching" or "qa",
        "timestamp": "ISO timestamp"
    }
    """
    await manager.connect(websocket, client_id)
    
    # Send welcome message (with error handling)
    try:
        await manager.send_message(client_id, {
            "type": "system",
            "message": "Connected to Interactive Teaching Agent (History Class 10)",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"Error sending welcome message to {client_id}: {str(e)}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                
                # Validate message format
                if message_data.get("type") != "message":
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Invalid message type. Expected 'message'",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                user_message = message_data.get("content", "").strip()
                
                if not user_message:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Empty message received",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                # Get teaching agent for this client
                agent = manager.get_agent(client_id)
                
                if not agent:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Agent not initialized",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                # Check if audio was interrupted (user sent new message while audio was playing)
                audio_interrupted = message_data.get("audio_interrupted", False)
                
                # If in teaching mode and audio was interrupted, pause teaching
                if audio_interrupted and agent.current_mode == "teaching":
                    agent.teaching_context["paused"] = True
                    # Save the last teaching point if available
                    if len(agent.chat_history) >= 2:
                        for msg in reversed(agent.chat_history[:-1]):
                            if msg["role"] == "assistant":
                                agent.teaching_context["last_teaching_point"] = msg["content"]
                                break
                    print(f"Teaching paused due to audio interruption for {client_id}")
                
                # Process message through teaching agent
                print(f"Processing message from {client_id}: {user_message[:50]}...")
                response = agent.process_message(user_message)
                
                # Prepare metadata
                metadata = {
                    "mode": agent.current_mode,
                    "current_topic": agent.current_topic,
                    "next_topic": agent.next_topic,
                    "topics_to_teach": agent.topics_to_teach,
                    "topics_covered": agent.teaching_context.get("topics_covered", []),
                    "current_topic_complete": agent.current_topic_complete,
                }
                
                # Stream response with TTS
                await stream_response_with_tts(websocket, response, metadata)
                
            except json.JSONDecodeError:
                try:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception:
                    # Connection might be closed
                    pass
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                try:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": f"Error processing message: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception:
                    # Connection might be closed
                    pass
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error for {client_id}: {str(e)}")
        manager.disconnect(client_id)


if __name__ == "__main__":
    print("=" * 80)
    print("Starting Interactive Teaching Agent WebSocket Server")
    print("=" * 80)
    print(f"Subject: History")
    print(f"Class Level: 10")
    print(f"Chapter: A Brief History of India")
    print(f"\nEndpoints:")
    print(f"  - WebSocket: ws://localhost:8000/ws/{{client_id}}")
    print(f"  - Demo Client: http://localhost:8000/demo")
    print(f"  - Health Check: http://localhost:8000/health")
    print("=" * 80)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

