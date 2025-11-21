# WebSocket Server - Quick Start Guide

## Installation

### 1. Install Required Packages

```bash
cd /Users/aprameyar/Developer/TutionMaster/AI/Langchains

# Install new dependencies (FastAPI, Uvicorn, WebSockets)
pip install fastapi uvicorn[standard] websockets python-multipart
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Starting the Server

### Method 1: Direct Python

```bash
python3 app.py
```

### Method 2: Using Uvicorn Command

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
================================================================================
Starting Interactive Teaching Agent WebSocket Server
================================================================================
Subject: History
Class Level: 10
Chapter: A Brief History of India

Endpoints:
  - WebSocket: ws://localhost:8000/ws/{client_id}
  - Demo Client: http://localhost:8000/demo
  - Health Check: http://localhost:8000/health
================================================================================
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Testing the Server

### 1. Quick Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
    "status": "healthy",
    "active_connections": 0,
    "timestamp": "2025-11-21T10:30:00.000Z"
}
```

### 2. Web Demo Client

Open your browser:
```
http://localhost:8000/demo
```

You'll see an interactive chat interface to test the teaching agent.

### 3. Python Client

```bash
python3 websocket_client_example.py
```

For automated demo:
```bash
python3 websocket_client_example.py demo
```

### 4. API Information

```bash
curl http://localhost:8000/
```

## Example Conversation

1. **Open demo client:** `http://localhost:8000/demo`

2. **Type messages:**
   - "teach me about ancient India"
   - "tell me more"
   - "what year was the Indus Valley discovered?"
   - "continue"

3. **Watch responses** appear in real-time!

## Directory Structure

```
AI/Langchains/
├── app.py                          # FastAPI WebSocket server
├── teaching_agent.py               # LangChain-based teaching agent
├── interactive_tutor.py            # Teaching tool
├── tutionmaster_faq.py             # Q&A tool
├── websocket_client_example.py    # Python client example
├── requirements.txt                # Updated with FastAPI, uvicorn
├── WEBSOCKET_API.md                # Full API documentation
└── QUICKSTART_WEBSOCKET.md         # This file
```

## Troubleshooting

### Port Already in Use

If port 8000 is in use:
```bash
# Use different port
uvicorn app:app --port 8080
```

### Module Not Found

Install missing packages:
```bash
pip install fastapi uvicorn websockets
```

### Connection Refused

1. Check if server is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check firewall settings

3. Verify ChromaDB is accessible

### Agent Not Responding

1. Check environment variables are set:
   ```bash
   cat .env
   ```

2. Verify ChromaDB has data:
   ```bash
   python3 test_system_setup.py
   ```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

gunicorn app:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t teaching-agent .
docker run -p 8000:8000 teaching-agent
```

### Using Systemd Service

Create `/etc/systemd/system/teaching-agent.service`:
```ini
[Unit]
Description=Interactive Teaching Agent WebSocket Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/AI/Langchains
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable teaching-agent
sudo systemctl start teaching-agent
```

## Configuration

### Environment Variables

Create or update `.env`:
```env
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=your-endpoint-here
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### Server Settings

In `app.py`, modify:
```python
uvicorn.run(
    "app:app",
    host="0.0.0.0",     # Listen address
    port=8000,          # Port number
    reload=True,        # Auto-reload (dev only)
    log_level="info"    # Logging level
)
```

### CORS Settings

For production, update allowed origins in `app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Client Examples

### JavaScript/Browser

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/user123');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Agent:', data.response);
};

ws.send(JSON.stringify({
    type: 'message',
    content: 'teach me about ancient India'
}));
```

### Python

```python
import asyncio
import websockets
import json

async def main():
    async with websockets.connect('ws://localhost:8000/ws/user123') as ws:
        await ws.send(json.dumps({
            'type': 'message',
            'content': 'teach me'
        }))
        
        response = await ws.recv()
        print(json.loads(response))

asyncio.run(main())
```

### cURL (for testing)

```bash
# Install websocat
brew install websocat  # macOS
# or: cargo install websocat

# Connect and send message
echo '{"type":"message","content":"teach me"}' | websocat ws://localhost:8000/ws/test
```

## Monitoring

### Server Logs

Uvicorn logs show:
- Connection events
- Message processing
- Errors and warnings
- Performance metrics

### Active Connections

```bash
curl http://localhost:8000/health | jq
```

### Performance Testing

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Load test (WebSocket)
# Note: Standard HTTP load testing doesn't work well with WebSocket
# Use specialized tools like `websocket-bench`
```

## Next Steps

1. **Test locally** with demo client
2. **Integrate** with your frontend
3. **Deploy** to production server
4. **Monitor** connections and performance
5. **Scale** with load balancer if needed

## Support

For issues:
1. Check `WEBSOCKET_API.md` for full documentation
2. Run `python3 test_system_setup.py` to verify setup
3. Check server logs for errors

---

**Status:** ✅ Ready to use
**Server:** FastAPI + Uvicorn
**WebSocket:** Real-time bidirectional communication
**Demo:** Available at `/demo`

