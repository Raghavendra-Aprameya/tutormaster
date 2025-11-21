# ElevenLabs TTS Integration Setup

## Overview

The teaching agent now includes Text-to-Speech (TTS) functionality using ElevenLabs API. All LLM responses are automatically converted to speech and streamed to the frontend.

## Setup Instructions

### 1. Install Dependencies

```bash
cd AI/Langchains
pip install -r requirements.txt
```

This will install:
- `elevenlabs` - ElevenLabs Python SDK (optional, we use direct API calls)
- `aiohttp` - For async HTTP requests to ElevenLabs API

### 2. Get ElevenLabs API Key

1. Sign up at [ElevenLabs](https://elevenlabs.io/)
2. Get your API key from the dashboard
3. Add it to your `.env` file:

```env
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Optional: Default voice (Rachel)
```

### 3. Voice Selection

You can use any ElevenLabs voice ID. Some popular options:
- `21m00Tcm4TlvDq8ikWAM` - Rachel (default, female, clear)
- `AZnzlk1XvdvUeBnXmlld` - Domi (female, expressive)
- `EXAVITQu4vr4xnSDxMaL` - Bella (female, calm)
- `ErXwobaYiN019PkySvjV` - Antoni (male, clear)
- `MF3mGyEYCl7XYWbV9V6O` - Elli (female, friendly)

To find more voices, check the [ElevenLabs Voice Library](https://elevenlabs.io/voice-library).

## How It Works

### Backend Flow

1. **User sends message** → WebSocket receives it
2. **Agent processes** → `TeachingAgent.process_message()` generates response
3. **Text response sent** → Immediately sent to frontend as JSON
4. **TTS generation** → `generate_tts_audio()` calls ElevenLabs API
5. **Audio sent** → Audio bytes converted to base64 and sent as JSON

### Frontend Flow

1. **Text received** → Displayed immediately in chat
2. **Audio received** → Decoded from base64
3. **Audio queued** → Added to playback queue
4. **Auto-play** → Plays automatically when ready

## Message Format

### Text Response
```json
{
  "type": "response",
  "response": "The Indus Valley Civilization was...",
  "mode": "teaching",
  "current_topic": "Ancient India",
  "timestamp": "2025-01-21T10:30:00.000Z"
}
```

### Audio Response
```json
{
  "type": "audio",
  "audio": "base64_encoded_audio_data...",
  "format": "mp3",
  "timestamp": "2025-01-21T10:30:01.000Z"
}
```

## Features

- ✅ **Automatic TTS**: All agent responses are converted to speech
- ✅ **Streaming**: Text appears immediately, audio follows
- ✅ **Queue Management**: Multiple audio clips queue automatically
- ✅ **Error Handling**: Gracefully handles TTS failures
- ✅ **Visual Feedback**: Shows audio playback status

## Testing

1. Start the server:
```bash
python app.py
```

2. Open the demo client:
```
http://localhost:8000/demo
```

3. Send a message and you should see:
   - Text response appears immediately
   - Audio indicator shows "🔊 Playing audio..."
   - Audio plays automatically

## Troubleshooting

### No Audio Playing

1. **Check API Key**: Ensure `ELEVENLABS_API_KEY` is set in `.env`
2. **Check Console**: Look for error messages in browser console
3. **Check Network**: Verify ElevenLabs API is accessible
4. **Check Quota**: Ensure you have credits in your ElevenLabs account

### Audio Errors

- If you see "Audio Error" messages, check:
  - API key is valid
  - You have sufficient credits
  - Network connectivity to ElevenLabs

### Performance

- TTS generation takes 1-3 seconds depending on text length
- Audio is generated asynchronously, so text appears first
- Long responses may take longer to generate audio

## Customization

### Change Voice

Update `.env`:
```env
ELEVENLABS_VOICE_ID=your_preferred_voice_id
```

### Adjust Voice Settings

Edit `generate_tts_audio()` in `app.py`:
```python
"voice_settings": {
    "stability": 0.5,        # 0.0-1.0, higher = more stable
    "similarity_boost": 0.75  # 0.0-1.0, higher = more similar to voice
}
```

### Disable TTS

If you want to disable TTS temporarily, comment out the TTS call in `stream_response_with_tts()`:
```python
# audio_bytes = await generate_tts_audio(text)
```

## API Costs

ElevenLabs pricing:
- Free tier: 10,000 characters/month
- Paid plans: Varies by plan

Monitor your usage in the ElevenLabs dashboard.

