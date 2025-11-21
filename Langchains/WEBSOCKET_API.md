# WebSocket API Documentation

## Overview

The Interactive Teaching Agent is now exposed via **FastAPI WebSocket** for real-time bidirectional communication. This allows clients to interact with the teaching agent through WebSocket connections.

## Quick Start

### Start the Server

```bash
cd /Users/aprameyar/Developer/TutionMaster/AI/Langchains
python3 app.py
```

Server starts on: `http://localhost:8000`

### Test with Demo Client

Open browser: `http://localhost:8000/demo`

The demo client provides a simple web interface to test the WebSocket connection.

## API Endpoints

### 1. WebSocket Connection

**Endpoint:** `ws://localhost:8000/ws/{client_id}`

**Parameters:**
- `client_id`: Unique identifier for the client connection (e.g., "user_123")

**Features:**
- Each client gets a separate teaching agent instance
- Maintains conversation history per connection
- Supports concurrent connections

### 2. REST Endpoints

#### Root
```
GET /
```
Returns API information and status.

#### Health Check
```
GET /health
```
Returns server health status and active connection count.

#### Demo Client
```
GET /demo
```
Returns HTML demo client for testing.

## Message Format

### Client → Server

```json
{
    "type": "message",
    "content": "teach me about ancient India"
}
```

**Fields:**
- `type`: Must be "message"
- `content`: User's message text

### Server → Client

#### Success Response
```json
{
    "type": "response",
    "response": "Let me teach you about ancient India...",
    "mode": "teaching",
    "current_topic": "Indus Valley Civilization",
    "timestamp": "2025-11-21T10:30:00.000Z"
}
```

**Fields:**
- `type`: "response"
- `response`: Agent's response text
- `mode`: Current mode ("teaching" or "qa" or null)
- `current_topic`: Currently teaching topic (if in teaching mode)
- `timestamp`: ISO 8601 timestamp

#### Error Response
```json
{
    "type": "error",
    "message": "Error description",
    "timestamp": "2025-11-21T10:30:00.000Z"
}
```

#### System Message
```json
{
    "type": "system",
    "message": "Connected to Interactive Teaching Agent",
    "timestamp": "2025-11-21T10:30:00.000Z"
}
```

## Python Client Example

```python
import asyncio
import websockets
import json

async def chat_with_agent():
    client_id = "python_client_001"
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to teaching agent!")
        
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Server: {welcome}")
        
        # Send teaching request
        await websocket.send(json.dumps({
            "type": "message",
            "content": "teach me about ancient India"
        }))
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"\nAgent: {data['response']}")
        print(f"Mode: {data.get('mode')}")
        
        # Continue conversation
        await websocket.send(json.dumps({
            "type": "message",
            "content": "tell me more about the Indus Valley"
        }))
        
        response = await websocket.recv()
        data = json.loads(response)
        print(f"\nAgent: {data['response']}")

# Run the client
asyncio.run(chat_with_agent())
```

## JavaScript Client Example

```javascript
const clientId = 'js_client_001';
const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);

ws.onopen = function(event) {
    console.log('Connected!');
    
    // Send message
    ws.send(JSON.stringify({
        type: 'message',
        content: 'teach me about ancient India'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'response') {
        console.log('Agent:', data.response);
        console.log('Mode:', data.mode);
    } else if (data.type === 'error') {
        console.error('Error:', data.message);
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    console.log('Disconnected');
};
```

## Features

### 1. **Per-Client Agent Instances**
Each WebSocket connection gets its own teaching agent instance with:
- Separate conversation history
- Independent teaching context
- Isolated state management

### 2. **Real-Time Communication**
- Bidirectional messaging
- Instant responses
- No polling required

### 3. **Conversation Context**
- Full chat history maintained per connection
- Pause-answer-resume flow works across messages
- Teaching mode persists throughout connection

### 4. **Concurrent Connections**
- Multiple clients can connect simultaneously
- Each client is isolated
- Scalable architecture

### 5. **Error Handling**
- Graceful disconnection
- Error messages sent to client
- Connection recovery support

## Connection Lifecycle

```
Client                          Server
  |                               |
  |--- Connect (WebSocket) ------>|
  |<--- Accept Connection --------|
  |<--- Welcome Message ----------|
  |                               |
  |--- User Message ------------->|
  |                  (Process with TeachingAgent)
  |<--- Agent Response -----------|
  |                               |
  |--- Another Message ---------->|
  |<--- Agent Response -----------|
  |                               |
  |--- Disconnect --------------->|
  |                  (Cleanup resources)
```

## Configuration

### Server Settings

In `app.py`:
```python
uvicorn.run(
    "app:app",
    host="0.0.0.0",  # Listen on all interfaces
    port=8000,       # Port number
    reload=True,     # Auto-reload on code changes
    log_level="info" # Logging level
)
```

### CORS Settings

Currently allows all origins:
```python
allow_origins=["*"]  # Change in production!
```

For production, specify allowed origins:
```python
allow_origins=["https://yourdomain.com"]
```

## Production Deployment

### Using Uvicorn Directly

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn with Uvicorn Workers

```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables

Make sure `.env` file is in place with:
```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Testing

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Root Endpoint
```bash
curl http://localhost:8000/
```

### 3. WebSocket with wscat
```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c ws://localhost:8000/ws/test_client

# Send message
{"type": "message", "content": "teach me"}
```

### 4. Demo Client
Open browser: `http://localhost:8000/demo`

## Monitoring

### Active Connections
```bash
curl http://localhost:8000/health
```

Response:
```json
{
    "status": "healthy",
    "active_connections": 3,
    "timestamp": "2025-11-21T10:30:00.000Z"
}
```

### Server Logs
The server logs show:
- Connection events
- Message processing
- Errors and exceptions
- Active connection count

## Error Handling

### Common Errors

**Invalid JSON:**
```json
{
    "type": "error",
    "message": "Invalid JSON format"
}
```

**Empty Message:**
```json
{
    "type": "error",
    "message": "Empty message received"
}
```

**Agent Error:**
```json
{
    "type": "error",
    "message": "Error processing message: [details]"
}
```

## Example Conversation Flow

```
# Connect
Client: <connects to ws://localhost:8000/ws/user123>
Server: {"type": "system", "message": "Connected..."}

# Start teaching
Client: {"type": "message", "content": "teach me about India"}
Server: {
    "type": "response",
    "response": "Let's explore ancient India. We'll cover...",
    "mode": "teaching"
}

# Continue teaching
Client: {"type": "message", "content": "tell me more"}
Server: {
    "type": "response",
    "response": "The Indus Valley Civilization...",
    "mode": "teaching"
}

# Ask specific question (pause-answer-resume)
Client: {"type": "message", "content": "what year was this?"}
Server: {
    "type": "response",
    "response": "Around 2500 BCE. Good question! Continuing with...",
    "mode": "teaching"
}
```

## Scaling Considerations

### Horizontal Scaling
- Use Redis for shared session storage
- Implement sticky sessions for WebSocket connections
- Use load balancer with WebSocket support

### Vertical Scaling
- Increase uvicorn workers
- Optimize LLM response time
- Cache frequently accessed chunks

## Security Considerations

### Production Checklist
- [ ] Implement authentication (JWT, OAuth)
- [ ] Rate limiting per client
- [ ] Input validation and sanitization
- [ ] Secure WebSocket (WSS)
- [ ] CORS whitelist specific domains
- [ ] Monitor and log suspicious activity
- [ ] Implement connection timeouts

### Authentication Example
```python
async def authenticate_client(client_id: str, token: str):
    # Verify JWT token
    # Return user info or raise exception
    pass
```

## Troubleshooting

### Connection Refused
- Check if server is running
- Verify port 8000 is not in use
- Check firewall settings

### Agent Not Responding
- Check ChromaDB is accessible
- Verify environment variables
- Check server logs for errors

### Slow Responses
- LLM processing takes 2-5 seconds
- ChromaDB query adds 0.5-1 second
- Network latency varies

## Summary

✅ **FastAPI WebSocket** server running
✅ **Real-time** bidirectional communication
✅ **Per-client** teaching agent instances
✅ **Full features** including pause-answer-resume
✅ **Demo client** for quick testing
✅ **Production-ready** with proper error handling

---

**Server:** FastAPI + Uvicorn
**WebSocket Endpoint:** `ws://localhost:8000/ws/{client_id}`
**Demo:** `http://localhost:8000/demo`
**Status:** ✅ Ready for use

