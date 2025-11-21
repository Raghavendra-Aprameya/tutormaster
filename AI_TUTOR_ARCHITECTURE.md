
# AI Tutor Application - Architecture & Implementation Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Features](#core-features)
3. [Technology Stack](#technology-stack)
4. [Database Design](#database-design)
5. [Feature 1: Chapter-based Teaching](#feature-1-chapter-based-teaching)
6. [Feature 2: Q&A from Database](#feature-2-qa-from-database)
7. [Backend Architecture](#backend-architecture)
8. [AI/LLM Integration](#aillm-integration)
9. [Implementation Flow](#implementation-flow)
10. [Best Practices & Considerations](#best-practices--considerations)

---

## System Overview

The AI Tutor application is designed to provide personalized learning experiences through two main modes:

1. **Teaching Mode**: Context-aware tutoring based on teacher-uploaded content
2. **Q&A Mode**: Quick answers to general questions using a knowledge database

### High-Level Architecture

```
┌─────────────┐
│   Frontend  │
│  (ChatGPT   │
│  Interface) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      Backend API Layer          │
│  - Request Router               │
│  - Authentication               │
│  - Session Management           │
└──────┬──────────────────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌─────────────┐
│   AI Engine  │   │  Database   │
│   (LangChain)│   │ (PostgreSQL)│
└──────┬───────┘   └─────────────┘
       │
       ▼
┌──────────────┐
│ Azure OpenAI │
│   / OpenAI   │
└──────────────┘
```

---

## Core Features

### Feature 1: Chapter-based Teaching (Context-Aware Mode)
- Teacher uploads chapter content (PDF, text, markdown)
- Content is processed and stored as embeddings
- AI provides explanations, examples, and answers within the chapter's scope
- Maintains conversation context and adapts to student's understanding

### Feature 2: Q&A from Database (Retrieval Mode)
- Pre-populated knowledge base with common questions and answers
- Fast retrieval without LLM calls for basic questions
- Semantic search to find relevant answers
- Falls back to LLM if no suitable answer found

---

## Technology Stack

### Backend
- **Framework**: FastAPI / Flask (Python)
- **AI Framework**: LangChain
- **LLM Provider**: Azure OpenAI / OpenAI GPT-4
- **Vector Store**: PostgreSQL with pgvector extension
- **ORM**: SQLAlchemy
- **Task Queue**: Celery (for async document processing)

### Database
- **Primary DB**: PostgreSQL 15+
- **Extensions**: pgvector (for embeddings), pg_trgm (for text search)
- **Caching**: Redis (for session management and frequent queries)

### AI/ML
- **Embeddings**: text-embedding-ada-002 or Azure OpenAI embeddings
- **LLM**: GPT-4 / GPT-4-turbo
- **Document Processing**: LangChain Document Loaders, Text Splitters

---

## Database Design

### Entity Relationship Diagram

```
┌─────────────────┐
│     users       │
├─────────────────┤
│ id (PK)         │
│ email           │
│ role (enum)     │  teacher/student
│ created_at      │
└────────┬────────┘
         │
         │ 1:N
         ▼
┌─────────────────┐
│    subjects     │
├─────────────────┤
│ id (PK)         │
│ teacher_id (FK) │
│ name            │
│ description     │
│ created_at      │
└────────┬────────┘
         │
         │ 1:N
         ▼
┌─────────────────────┐
│     chapters        │
├─────────────────────┤
│ id (PK)             │
│ subject_id (FK)     │
│ title               │
│ content_text        │
│ file_path           │
│ status              │  processing/ready/failed
│ created_at          │
└────────┬────────────┘
         │
         │ 1:N
         ▼
┌──────────────────────┐
│  chapter_embeddings  │
├──────────────────────┤
│ id (PK)              │
│ chapter_id (FK)      │
│ chunk_text           │
│ embedding (vector)   │  pgvector type
│ chunk_index          │
│ metadata (JSONB)     │
└──────────────────────┘

┌─────────────────────┐
│  knowledge_base     │
├─────────────────────┤
│ id (PK)             │
│ question            │
│ answer              │
│ category            │
│ embedding (vector)  │  pgvector type
│ usage_count         │
│ created_at          │
└─────────────────────┘

┌─────────────────────┐
│  chat_sessions      │
├─────────────────────┤
│ id (PK)             │
│ user_id (FK)        │
│ chapter_id (FK)     │  nullable
│ mode (enum)         │  teaching/qa
│ created_at          │
│ last_active_at      │
└────────┬────────────┘
         │
         │ 1:N
         ▼
┌─────────────────────┐
│   chat_messages     │
├─────────────────────┤
│ id (PK)             │
│ session_id (FK)     │
│ role (enum)         │  user/assistant/system
│ content             │
│ tokens_used         │
│ created_at          │
└─────────────────────┘
```

### Key Tables Explained

#### `chapter_embeddings`
- Stores vector embeddings of chapter content chunks
- Uses pgvector extension for similarity search
- Each chunk is ~500-1000 tokens with overlap
- Metadata includes: page number, section title, source file

#### `knowledge_base`
- Pre-populated Q&A pairs
- Embeddings for semantic search
- Tracks usage for analytics and improvement

#### `chat_sessions`
- Maintains conversation context
- Links to specific chapter (teaching mode) or null (Q&A mode)
- Stores mode to determine routing logic

---

## Feature 1: Chapter-based Teaching

### Implementation Approach

#### 1. Content Upload & Processing

```python
# Flow: Teacher uploads chapter content

1. Teacher uploads PDF/DOCX/TXT file
2. Backend validates file and creates chapter record
3. Async task processes the document:
   a. Extract text from document
   b. Split text into chunks (LangChain TextSplitter)
   c. Generate embeddings for each chunk
   d. Store chunks and embeddings in database
4. Update chapter status to 'ready'
```

#### 2. Teaching Session Flow

```
User asks question about chapter
         ↓
Backend receives request with:
- session_id
- chapter_id  
- user_message
         ↓
Retrieval Augmented Generation (RAG):
1. Convert user question to embedding
2. Semantic search in chapter_embeddings
   (cosine similarity using pgvector)
3. Retrieve top-k relevant chunks (k=3-5)
         ↓
4. Build LLM prompt:
   - System: "You are a tutor for [Subject]"
   - Context: Retrieved chapter chunks
   - Chat history: Last N messages
   - User query
         ↓
5. Call LLM (GPT-4 via LangChain)
         ↓
6. Store response in chat_messages
7. Return response to frontend
```

#### 3. Code Structure

```python
# services/teaching_service.py

class TeachingService:
    def __init__(self):
        self.llm = AzureChatOpenAI(...)
        self.embeddings = OpenAIEmbeddings(...)
        self.vector_store = PGVector(...)
    
    async def process_chapter(self, chapter_id: int, file_path: str):
        """Process uploaded chapter content"""
        # Load document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings and store
        for idx, chunk in enumerate(chunks):
            embedding = await self.embeddings.aembed_query(chunk.page_content)
            await self.db.store_chunk_embedding(
                chapter_id=chapter_id,
                text=chunk.page_content,
                embedding=embedding,
                chunk_index=idx,
                metadata=chunk.metadata
            )
    
    async def chat(self, session_id: int, user_message: str):
        """Handle teaching mode chat"""
        # Get session context
        session = await self.db.get_session(session_id)
        chapter_id = session.chapter_id
        
        # Retrieve relevant chunks using semantic search
        query_embedding = await self.embeddings.aembed_query(user_message)
        relevant_chunks = await self.vector_store.similarity_search(
            query_embedding,
            chapter_id=chapter_id,
            k=5
        )
        
        # Get chat history
        history = await self.db.get_chat_history(session_id, limit=5)
        
        # Build prompt
        context = "\n\n".join([chunk.text for chunk in relevant_chunks])
        
        prompt = PromptTemplate(
            template="""You are an expert tutor teaching from the following content:
            
Context:
{context}

Conversation History:
{history}

Student Question: {question}

Provide a clear, educational response. Use examples from the context when relevant.
Encourage understanding over memorization."""
        )
        
        # Create chain
        chain = prompt | self.llm
        
        # Get response
        response = await chain.ainvoke({
            "context": context,
            "history": format_history(history),
            "question": user_message
        })
        
        # Store messages
        await self.db.store_message(session_id, "user", user_message)
        await self.db.store_message(session_id, "assistant", response.content)
        
        return response.content
```

---

## Feature 2: Q&A from Database

### Implementation Approach

#### 1. Knowledge Base Setup

```python
# Initial setup: Populate knowledge base

1. Admin/Teacher adds Q&A pairs to knowledge_base table
2. System generates embeddings for questions
3. Optional: Bulk import from CSV/JSON
```

#### 2. Q&A Flow

```
User asks general question
         ↓
Backend receives request
         ↓
1. Generate embedding for question
2. Semantic search in knowledge_base
   (cosine similarity threshold: 0.85)
         ↓
3a. High confidence match found (>0.85)?
    → Return stored answer directly
    
3b. Medium confidence (0.70-0.85)?
    → Use stored answer + LLM refinement
    
3c. Low confidence (<0.70)?
    → Fall back to LLM with general prompt
         ↓
4. Increment usage_count for matched Q&A
5. Return response to frontend
```

#### 3. Code Structure

```python
# services/qa_service.py

class QAService:
    def __init__(self):
        self.llm = AzureChatOpenAI(...)
        self.embeddings = OpenAIEmbeddings(...)
        self.similarity_threshold_high = 0.85
        self.similarity_threshold_med = 0.70
    
    async def answer_question(self, user_question: str, session_id: int):
        """Handle Q&A mode"""
        # Generate question embedding
        query_embedding = await self.embeddings.aembed_query(user_question)
        
        # Search knowledge base
        matches = await self.db.semantic_search_kb(
            embedding=query_embedding,
            limit=3
        )
        
        if not matches:
            return await self._fallback_llm(user_question, session_id)
        
        best_match = matches[0]
        similarity = best_match.similarity_score
        
        # High confidence - return directly
        if similarity >= self.similarity_threshold_high:
            await self.db.increment_usage(best_match.id)
            await self.db.store_message(session_id, "user", user_question)
            await self.db.store_message(session_id, "assistant", best_match.answer)
            return {
                "answer": best_match.answer,
                "source": "knowledge_base",
                "confidence": similarity
            }
        
        # Medium confidence - refine with LLM
        elif similarity >= self.similarity_threshold_med:
            refined_answer = await self._refine_with_llm(
                user_question, 
                best_match.answer
            )
            await self.db.increment_usage(best_match.id)
            return {
                "answer": refined_answer,
                "source": "kb_refined",
                "confidence": similarity
            }
        
        # Low confidence - fallback to LLM
        else:
            return await self._fallback_llm(user_question, session_id)
    
    async def _refine_with_llm(self, question: str, base_answer: str):
        """Refine KB answer with LLM"""
        prompt = f"""Based on this reference answer, respond to the user's question:

Reference: {base_answer}

User Question: {question}

Provide a natural, conversational response that addresses their specific question."""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _fallback_llm(self, question: str, session_id: int):
        """Fallback to general LLM"""
        history = await self.db.get_chat_history(session_id, limit=5)
        
        prompt = f"""You are a helpful educational assistant.

Conversation History:
{format_history(history)}

Student Question: {question}

Provide a clear, accurate answer."""
        
        response = await self.llm.ainvoke(prompt)
        
        # Store for future knowledge base population
        await self.db.store_potential_qa(question, response.content)
        
        return {
            "answer": response.content,
            "source": "llm",
            "confidence": None
        }
```

---

## Backend Architecture

### API Endpoints

```python
# api/routes/chat.py

from fastapi import APIRouter, Depends
from services.teaching_service import TeachingService
from services.qa_service import QAService

router = APIRouter()

@router.post("/api/v1/chat/start")
async def start_session(
    user_id: int,
    mode: str,  # "teaching" or "qa"
    chapter_id: Optional[int] = None
):
    """Initialize a new chat session"""
    session = await db.create_session(
        user_id=user_id,
        mode=mode,
        chapter_id=chapter_id
    )
    return {"session_id": session.id}

@router.post("/api/v1/chat/message")
async def send_message(
    session_id: int,
    message: str,
    teaching_service: TeachingService = Depends(),
    qa_service: QAService = Depends()
):
    """Send a message in the chat"""
    session = await db.get_session(session_id)
    
    if session.mode == "teaching":
        if not session.chapter_id:
            raise HTTPException(400, "Chapter required for teaching mode")
        response = await teaching_service.chat(session_id, message)
    else:  # qa mode
        response = await qa_service.answer_question(message, session_id)
    
    return response

@router.post("/api/v1/chapters/upload")
async def upload_chapter(
    subject_id: int,
    title: str,
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    """Upload and process a new chapter"""
    # Save file
    file_path = await save_upload(file)
    
    # Create chapter record
    chapter = await db.create_chapter(
        subject_id=subject_id,
        title=title,
        file_path=file_path,
        status="processing"
    )
    
    # Process asynchronously
    background_tasks.add_task(
        teaching_service.process_chapter,
        chapter.id,
        file_path
    )
    
    return {"chapter_id": chapter.id, "status": "processing"}

@router.get("/api/v1/chapters/{chapter_id}/status")
async def get_chapter_status(chapter_id: int):
    """Check if chapter is ready for teaching"""
    chapter = await db.get_chapter(chapter_id)
    return {"status": chapter.status}

@router.post("/api/v1/knowledge-base/add")
async def add_qa_pair(
    question: str,
    answer: str,
    category: str
):
    """Add a new Q&A pair to knowledge base"""
    embedding = await embeddings.aembed_query(question)
    
    await db.add_to_knowledge_base(
        question=question,
        answer=answer,
        category=category,
        embedding=embedding
    )
    
    return {"status": "success"}
```

### Request Router Logic

```python
# middleware/router.py

class ChatRouter:
    """Routes requests to appropriate service"""
    
    async def route(self, session_id: int, message: str):
        session = await db.get_session(session_id)
        
        # Teaching mode: Use chapter-specific context
        if session.mode == "teaching":
            return await self.teaching_service.chat(session_id, message)
        
        # Q&A mode: Check knowledge base first
        elif session.mode == "qa":
            return await self.qa_service.answer_question(message, session_id)
        
        else:
            raise ValueError(f"Unknown mode: {session.mode}")
```

---

## AI/LLM Integration

### LangChain Setup

```python
# config/ai_config.py

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

# Initialize LLM
llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_deployment="gpt-4o-mini",
    temperature=0.7,  # Slightly creative for teaching
    max_tokens=1000
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002"
)

# Initialize vector store
vector_store = PGVector(
    connection_string="postgresql://...",
    embedding_function=embeddings,
    collection_name="chapter_embeddings"
)
```

### Prompt Engineering

#### Teaching Mode Prompts

```python
TEACHING_SYSTEM_PROMPT = """You are an expert tutor specializing in {subject}.

Your teaching style:
- Break down complex concepts into simple explanations
- Use relevant examples and analogies
- Ask guiding questions to check understanding
- Encourage critical thinking
- Adapt to the student's level of knowledge

Always base your explanations on the provided context from the chapter.
If a question is outside the scope of the current chapter, politely redirect to the chapter content.
"""

TEACHING_USER_PROMPT = """Context from Chapter:
{context}

Recent Conversation:
{history}

Student's Question: {question}

Provide a clear, educational response."""
```

#### Q&A Mode Prompts

```python
QA_SYSTEM_PROMPT = """You are a helpful educational assistant.

Provide clear, concise, and accurate answers.
If you're not certain about something, acknowledge it.
Always aim to educate and help the student learn."""

QA_REFINEMENT_PROMPT = """You have this reference answer: {reference}

The student asked: {question}

Provide a natural response that addresses their specific question while incorporating information from the reference."""
```

---

## Implementation Flow

### Phase 1: Setup & Infrastructure (Week 1-2)

1. **Database Setup**
   - Install PostgreSQL with pgvector extension
   - Create database schema
   - Set up connection pooling

2. **Backend Foundation**
   - Set up FastAPI application
   - Implement authentication & authorization
   - Create basic CRUD operations

3. **AI Integration**
   - Configure Azure OpenAI / OpenAI credentials
   - Set up LangChain components
   - Test embedding generation

### Phase 2: Teaching Mode (Week 3-4)

1. **Document Processing Pipeline**
   - Implement file upload handling
   - Create document parsing (PDF, DOCX, TXT)
   - Build chunking & embedding logic
   - Set up async task queue (Celery)

2. **RAG Implementation**
   - Implement semantic search in pgvector
   - Build context retrieval logic
   - Create prompt templates
   - Integrate LLM calls

3. **Chat Interface**
   - Implement session management
   - Build chat message storage
   - Create conversation history handling

### Phase 3: Q&A Mode (Week 5)

1. **Knowledge Base**
   - Create admin interface for Q&A management
   - Implement bulk import functionality
   - Build semantic search for KB

2. **Hybrid Answering Logic**
   - Implement confidence-based routing
   - Create LLM refinement logic
   - Build fallback mechanisms

### Phase 4: Frontend Integration (Week 6-7)

1. **Chat Interface**
   - Build ChatGPT-like UI
   - Implement mode switching
   - Add typing indicators & streaming responses

2. **Teacher Dashboard**
   - Chapter upload & management
   - Processing status monitoring
   - Knowledge base management

3. **Student Interface**
   - Chapter selection
   - Chat history
   - Progress tracking

### Phase 5: Optimization & Testing (Week 8)

1. **Performance**
   - Implement caching (Redis)
   - Optimize database queries
   - Add response streaming

2. **Quality**
   - Test with various content types
   - Refine prompts based on testing
   - A/B test different models

---

## Best Practices & Considerations

### 1. Cost Optimization

```python
# Implement token tracking
class TokenTracker:
    def __init__(self):
        self.llm = llm
        
    async def call_with_tracking(self, prompt: str, user_id: int):
        response = await self.llm.ainvoke(prompt)
        
        # Log token usage
        await db.log_token_usage(
            user_id=user_id,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=calculate_cost(response.usage)
        )
        
        return response

# Use smaller models when appropriate
# GPT-4o-mini for Q&A, GPT-4 for complex teaching
```

### 2. Quality Assurance

```python
# Implement response evaluation
class ResponseEvaluator:
    async def evaluate(self, question: str, response: str, context: str):
        # Check for hallucinations
        faithfulness = await self.check_faithfulness(response, context)
        
        # Check relevance
        relevance = await self.check_relevance(question, response)
        
        # Log poor responses for review
        if faithfulness < 0.7 or relevance < 0.7:
            await db.flag_for_review(question, response, context)
```

### 3. Security

- **Input Validation**: Sanitize all user inputs
- **Rate Limiting**: Prevent abuse (10 messages/minute per user)
- **Authentication**: JWT tokens for API access
- **Data Privacy**: Encrypt sensitive data at rest
- **Content Filtering**: Block inappropriate queries

### 4. Scalability

```python
# Use connection pooling
DATABASE_URL = "postgresql+asyncpg://..."
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40
)

# Implement caching for frequent queries
@cache(ttl=3600)
async def get_chapter_embeddings(chapter_id: int):
    return await db.fetch_embeddings(chapter_id)

# Use async everywhere
async def process_batch(chapters: List[Chapter]):
    tasks = [process_chapter(ch.id) for ch in chapters]
    await asyncio.gather(*tasks)
```

### 5. Monitoring & Analytics

```python
# Track key metrics
metrics = {
    "response_time": [],
    "token_usage": [],
    "user_satisfaction": [],
    "retrieval_accuracy": [],
    "fallback_rate": []
}

# Implement logging
import structlog
logger = structlog.get_logger()

logger.info(
    "chat_message",
    session_id=session_id,
    mode=mode,
    response_time=elapsed,
    tokens_used=tokens,
    source=source  # kb/llm/kb_refined
)
```

### 6. Content Updates

```python
# Version control for chapters
class ChapterVersion:
    async def update_chapter(self, chapter_id: int, new_content: str):
        # Create new version
        version = await db.create_chapter_version(
            chapter_id=chapter_id,
            content=new_content,
            version_number=current_version + 1
        )
        
        # Reprocess embeddings
        await teaching_service.process_chapter(chapter_id, new_content)
        
        # Mark old embeddings as archived
        await db.archive_old_embeddings(chapter_id, version_number)
```

### 7. Fallback Strategies

```python
# Multiple fallback layers
async def get_answer(question: str):
    try:
        # Try primary LLM
        return await primary_llm.call(question)
    except Exception as e:
        logger.error("Primary LLM failed", error=e)
        
        try:
            # Fallback to secondary model
            return await fallback_llm.call(question)
        except Exception as e2:
            logger.error("Fallback LLM failed", error=e2)
            
            # Return cached response or error message
            return "I'm having trouble right now. Please try again."
```

---

## Deployment Checklist

- [ ] Set up PostgreSQL with pgvector
- [ ] Configure Azure OpenAI credentials
- [ ] Set up Redis for caching
- [ ] Implement rate limiting
- [ ] Set up logging and monitoring
- [ ] Create admin dashboard
- [ ] Write API documentation
- [ ] Set up CI/CD pipeline
- [ ] Configure backup strategy
- [ ] Load test the system
- [ ] Create user documentation

---

## Future Enhancements

1. **Multi-modal Support**: Images, diagrams, equations in chapters
2. **Voice Interface**: Speech-to-text for questions
3. **Assessment**: Quiz generation from chapter content
4. **Personalization**: Adaptive learning based on student performance
5. **Collaborative Learning**: Group study sessions
6. **Mobile App**: Native iOS/Android apps
7. **Analytics Dashboard**: Detailed insights for teachers
8. **Multi-language Support**: Translate content and responses

---

## Resources

- **LangChain Docs**: https://python.langchain.com/docs/
- **pgvector**: https://github.com/pgvector/pgvector
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenAI Best Practices**: https://platform.openai.com/docs/guides/prompt-engineering

---

## Contact & Support

For questions or issues, please refer to the project documentation or reach out to the development team.

**Last Updated**: November 2025
**Version**: 1.0

