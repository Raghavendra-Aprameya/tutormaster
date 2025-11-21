# Interactive Teaching System

An AI-powered agent that provides comprehensive interactive teaching sessions with the ability to handle student interruptions and questions seamlessly.

## Architecture

The system consists of three main components:

### 1. Teaching Agent (`teaching_agent.py`)
- **Main orchestrator** that maintains shared conversation history
- Routes messages to appropriate tools based on context and intent
- Implements **pause-answer-resume** flow for handling interruptions
- Manages teaching state and session context

### 2. Interactive Tutor Tool (`interactive_tutor.py`)
- **Stateless teaching tool** for comprehensive chapter teaching
- Extracts topics from chapters
- Generates educational content based on retrieved chunks
- Continues teaching conversations using chat history

### 3. RAG Q&A Tool (`tutionmaster_faq.py`)
- **Quick answer tool** for specific factual questions
- Retrieves relevant chunks from ChromaDB
- Generates concise answers with conversation context
- Updated to accept chat history for context-aware responses

## Key Features

### 🎓 Comprehensive Teaching
- Teaches entire chapters systematically
- Covers all topics with detailed explanations
- Uses examples from the source material
- Adapts to student's pace and understanding

### 💬 Conversational & Interactive
- Maintains full conversation history
- Natural back-and-forth dialogue
- Encourages questions and engagement
- Checks for understanding periodically

### ⏸️ Pause-Answer-Resume
When a student asks a specific question during teaching:
1. **PAUSE**: System stores current teaching point
2. **ANSWER**: Routes to RAG Q&A for quick factual answer
3. **RESUME**: Continues teaching from where it left off

### 🔄 Shared Conversation History
- Both tools access the same chat history
- Teaching tool knows about previous Q&A
- Q&A tool has context from teaching session
- Seamless transitions between modes

## Usage

### Interactive Mode (Command Line)

```bash
cd AI/Langchains
python teaching_agent.py
```

Start chatting with the agent:
- "teach me about ancient India" → Starts teaching session
- "Tell me more about cities" → Continues teaching
- "What year was this?" → Pauses, answers, resumes
- "exit" → Ends session

### Run Examples

```bash
python example_teaching_session.py
```

Choose from:
1. **Full Simulated Conversation** - Demonstrates all features
2. **Interactive Mode** - Chat with agent live
3. **Quick Demo** - Concise feature overview

### Programmatic Usage

```python
from teaching_agent import TeachingAgent

# Initialize agent
agent = TeachingAgent()

# Start teaching
response = agent.process_message("teach me about ancient India")
print(response)

# Continue conversation
response = agent.process_message("Tell me more about the Indus Valley")
print(response)

# Ask specific question (triggers pause-answer-resume)
response = agent.process_message("What year was it discovered?")
print(response)

# Access conversation history
print(f"Messages: {len(agent.chat_history)}")
print(f"Mode: {agent.current_mode}")
```

## How It Works

### Message Flow

```
User Message
    ↓
Agent (analyzes intent)
    ↓
    ├─→ Teaching Request? → start_teaching()
    ├─→ In Teaching Mode?
    │   ├─→ Specific Question? → pause_answer_resume()
    │   │   ├─→ Q&A Tool (answer)
    │   │   └─→ Teaching Tool (resume)
    │   └─→ General Response? → continue_teaching()
    └─→ Default → Q&A Tool
    ↓
Response + Update History
```

### Pause-Answer-Resume Flow

```
Teaching: "The Indus Valley had advanced cities..."
    ↓
Student: "What year was this?" [SPECIFIC QUESTION DETECTED]
    ↓
[PAUSE] Store: last_teaching_point = "advanced cities..."
    ↓
[ANSWER] RAG Q&A: "Around 2500 BCE..."
    ↓
[RESUME] Teaching: "Good question! Now continuing with city planning..."
    ↓
Teaching continues seamlessly
```

### Decision Logic

The agent routes to **Q&A Tool** if message contains:
- "what is", "what are", "when did", "where"
- "define", "who was", "how many"
- Short question with "?"
- Clear factual lookup intent

Otherwise, routes to **Teaching Tool** for:
- Teaching continuations
- Elaborations and clarifications
- Broader exploratory questions
- General responses

## Configuration

### Environment Variables

Required in `.env` file:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### ChromaDB Setup

The system expects:
- ChromaDB at: `../../chroma_store` (relative to Langchains folder)
- Collection name: `chapter_embeddings`
- Metadata fields: `subject`, `chapter`, `class_level`

## Teaching Context

The agent maintains:

```python
teaching_context = {
    "subject": "History",
    "chapter": "A Brief History of India",
    "class_level": "10",
    "topics_covered": ["Indus Valley", ...],
    "current_topic": "Vedic Period",
    "last_teaching_point": "...",
    "paused": False
}
```

## Testing

### Test Individual Components

```bash
# Test teaching tool standalone
python interactive_tutor.py

# Test Q&A tool
python tutionmaster_faq.py

# Test full agent
python teaching_agent.py
```

### Run Full Demo

```bash
python example_teaching_session.py
```

## Extending the System

### Add New Tool

1. Create tool class with stateless design
2. Add to `TeachingAgent.__init__()`:
   ```python
   self._new_tool = None
   
   @property
   def new_tool(self):
       if self._new_tool is None:
           from new_tool import NewTool
           self._new_tool = NewTool()
       return self._new_tool
   ```

3. Update routing logic in `process_message()`
4. Tool receives `chat_history` for context

### Modify Teaching Behavior

Edit prompts in `interactive_tutor.py`:
- `generate_initial_teaching()` - Opening message
- `continue_teaching()` - Ongoing teaching style
- Adjust temperature for more/less creativity

### Change Decision Logic

Modify `_is_specific_question()` in `teaching_agent.py`:
- Add/remove question indicators
- Adjust word count threshold
- Implement ML-based intent classification

## Files Structure

```
AI/Langchains/
├── teaching_agent.py              # Main orchestrator
├── interactive_tutor.py           # Teaching tool
├── tutionmaster_faq.py           # Q&A tool (updated)
├── example_teaching_session.py   # Comprehensive examples
└── TEACHING_SYSTEM_README.md     # This file
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `langchain`
- `langchain-openai`
- `langchain-chroma`
- `langchain-huggingface`
- `langchain-community`
- `python-dotenv`
- `chromadb`
- `sentence-transformers`

## Troubleshooting

### "No chunks retrieved"
- Check ChromaDB path is correct (`../../chroma_store`)
- Verify data is ingested using `create_embeddings.py`
- Check metadata filters match stored data

### "Content filter triggered"
- Some historical content may trigger Azure's content filters
- Adjust prompt to be more educational/neutral
- Use different query phrasing

### "Teaching not starting"
- Ensure trigger phrases are used: "teach me", "I want to learn"
- Check ChromaDB has data for the subject
- Verify environment variables are set

## Future Enhancements

- [ ] Add quiz/assessment tool
- [ ] Implement topic tracking and progress
- [ ] Add student knowledge profiling
- [ ] Support multiple concurrent sessions
- [ ] Add voice interaction support
- [ ] Implement learning path recommendations
- [ ] Add multimodal support (images, diagrams)

## License

Part of TutionMaster project.

