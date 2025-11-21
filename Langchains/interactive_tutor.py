import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Chroma configuration
CHROMA_DIR = "../../chroma_store"
MODEL_NAME = "BAAI/bge-small-en"
DEVICE = "cpu"
COLLECTION_NAME = "chapter_embeddings"


class InteractiveTutorTool:
    """
    Stateless teaching tool that receives context from the agent.
    Provides comprehensive interactive teaching for chapters.
    """
    
    def __init__(self):
        """Initialize the embedding model and ChromaDB connection"""
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": DEVICE}
        )
        
        # Initialize ChromaDB
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_DIR
        )
        
        # Initialize Azure OpenAI
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=openai_endpoint,
            api_version="2024-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.6,  # Slightly higher for more creative teaching
        )
    
    def get_available_chapters(self, subject: Optional[str] = None, 
                               class_level: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Query ChromaDB metadata to get list of available chapters.
        
        Args:
            subject: Optional subject filter
            class_level: Optional class level filter
            
        Returns:
            List of dictionaries with chapter information
        """
        try:
            # Get all documents from the collection
            collection = self.db._collection
            results = collection.get(include=["metadatas"])
            
            if not results or not results.get("metadatas"):
                return []
            
            # Extract unique combinations of subject, chapter, class_level
            chapters_set = set()
            for metadata in results["metadatas"]:
                if metadata:
                    subject_val = metadata.get("subject", "Unknown")
                    chapter_val = metadata.get("chapter", "Unknown")
                    class_level_val = metadata.get("class_level", "Unknown")
                    
                    # Apply filters if provided
                    if subject and subject != subject_val:
                        continue
                    if class_level and class_level != class_level_val:
                        continue
                    
                    chapters_set.add((subject_val, chapter_val, class_level_val))
            
            # Convert to list of dictionaries
            chapters = [
                {
                    "subject": subj,
                    "chapter": chap,
                    "class_level": cls_lvl
                }
                for subj, chap, cls_lvl in sorted(chapters_set)
            ]
            
            return chapters
            
        except Exception as e:
            print(f"Error getting available chapters: {str(e)}")
            return []
    
    def extract_topics(self, subject: str, chapter: str, class_level: str) -> List[str]:
        """
        Extract main topics from a chapter by analyzing its chunks.
        
        Args:
            subject: Subject name
            chapter: Chapter title
            class_level: Class level
            
        Returns:
            List of topic strings
        """
        try:
            # Retrieve sample chunks from the chapter (spread across pages)
            filter_meta = {
                "$and": [
                    {"subject": {"$eq": subject}},
                    {"chapter": {"$eq": chapter}},
                    {"class_level": {"$eq": class_level}}
                ]
            }
            
            # Get diverse chunks using a generic query
            results = self.db.similarity_search(
                "main topics concepts overview",
                k=20,  # Get more chunks to analyze
                filter=filter_meta
            )
            
            if not results:
                return ["General Introduction", "Key Concepts", "Conclusion"]
            
            # Sample chunks from different pages
            sampled_chunks = []
            seen_pages = set()
            for doc in results:
                page_num = doc.metadata.get("page_number")
                if page_num not in seen_pages:
                    sampled_chunks.append(doc.page_content)
                    seen_pages.add(page_num)
                if len(sampled_chunks) >= 10:
                    break
            
            # Combine chunks for analysis
            combined_text = "\n\n".join(sampled_chunks)
            
            # Use LLM to extract topics
            prompt = f"""Based on the following excerpts from a chapter titled "{chapter}" in {subject} for Class {class_level}, 
identify and list the main topics/concepts covered in this chapter.

Excerpts:
{combined_text[:4000]}  

Please list 4-7 main topics in a clear, structured format. Each topic should be a concise phrase.
Format: Just list the topics, one per line, without numbers or bullets."""
            
            response = self.llm.invoke(prompt)
            topics_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse topics from response
            topics = [
                line.strip().lstrip('-').lstrip('•').lstrip('*').strip()
                for line in topics_text.split('\n')
                if line.strip() and len(line.strip()) > 5
            ]
            
            # Limit to reasonable number
            topics = topics[:8]
            
            return topics if topics else ["Introduction", "Main Concepts", "Summary"]
            
        except Exception as e:
            print(f"Error extracting topics: {str(e)}")
            return ["Introduction", "Key Concepts", "Conclusion"]
    
    def load_chapter_chunks(self, subject: str, chapter: str, class_level: str) -> List[Dict]:
        """
        Load all chunks for a chapter into memory for faster access during teaching.
        
        Args:
            subject: Subject name
            chapter: Chapter title
            class_level: Class level
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        try:
            filter_meta = {
                "$and": [
                    {"subject": {"$eq": subject}},
                    {"chapter": {"$eq": chapter}},
                    {"class_level": {"$eq": class_level}}
                ]
            }
            
            # Use a broad query to get all chunks
            results = self.db.similarity_search(
                "comprehensive overview all topics",
                k=100,  # Get many chunks
                filter=filter_meta
            )
            
            chunks = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
            
            print(f"Loaded {len(chunks)} chunks for {chapter}")
            return chunks
            
        except Exception as e:
            print(f"Error loading chapter chunks: {str(e)}")
            return []
    
    def generate_initial_teaching(self, subject: str, chapter: str, class_level: str,
                                  topics: List[str], chapter_chunks: List[Dict]) -> str:
        """
        Generate the first teaching message introducing the chapter.
        
        Args:
            subject: Subject name
            chapter: Chapter title
            class_level: Class level
            topics: List of topics to cover
            chapter_chunks: Pre-loaded chapter chunks
            
        Returns:
            Initial teaching message
        """
        # Get relevant chunks for introduction
        relevant_chunks = self._get_relevant_chunks(
            chapter_chunks,
            "introduction overview beginning first concepts",
            k=6
        )
        
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        topics_list = "\n".join([f"- {topic}" for topic in topics])
        
        prompt = f"""You are an interactive tutor teaching {subject} (Class {class_level}) from a PDF textbook.

Chapter: {chapter}

Topics you will cover:
{topics_list}

Teaching Material from PDF:
{context[:3000]}

CRITICAL INSTRUCTIONS:
- Base ALL your teaching ONLY on the provided teaching material from the PDF
- NEVER use general knowledge or external information
- Use ONLY examples and facts present in the teaching material
- If something is not in the material, don't mention it

Your task:
1. Give a warm, engaging introduction to the chapter using content from the PDF
2. Briefly mention what topics we'll cover
3. Start explaining the first topic comprehensively using ONLY the provided material
4. Use examples from the PDF material when possible
5. Be encouraging and ask if the student is ready to begin

Keep your response comprehensive but not overwhelming (3-4 paragraphs). Make it feel like a real classroom interaction."""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def continue_teaching(self, chat_history: List[Dict], chapter_chunks: List[Dict],
                         user_message: str) -> str:
        """
        Continue teaching session with full context.
        
        Args:
            chat_history: Full conversation history from agent
            chapter_chunks: Pre-loaded chapter chunks
            user_message: Latest user message
            
        Returns:
            Teaching response
        """
        # Get relevant chunks based on recent conversation
        recent_context = " ".join([
            msg["content"] for msg in chat_history[-4:]
            if msg["role"] in ["user", "assistant"]
        ])
        
        relevant_chunks = self._get_relevant_chunks(
            chapter_chunks,
            recent_context + " " + user_message,
            k=8
        )
        
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        
        # Format chat history for LLM
        history_text = self._format_chat_history(chat_history)
        
        # Create teaching prompt
        prompt = f"""You are an interactive tutor teaching from a PDF textbook. 
Review the conversation history and continue teaching based on the student's response.

Conversation History:
{history_text}

Teaching Material from PDF (relevant excerpts):
{context[:4000]}

Student's latest message: {user_message}

CRITICAL INSTRUCTIONS:
- Use ONLY the provided teaching material from the PDF
- NEVER add information from general knowledge
- Base all explanations and examples strictly on the PDF excerpts provided
- If the student asks about something not in the material, politely say it's not covered in our textbook

Your response should:
1. Address the student's message appropriately using PDF content
2. Continue teaching comprehensively using ONLY the material provided
3. Provide detailed explanations with examples from the PDF excerpts
4. Check for understanding periodically
5. Be encouraging and maintain engagement
6. Progress through topics naturally

Continue the teaching conversation naturally, strictly based on the PDF material."""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _get_relevant_chunks(self, chunks: List[Dict], query: str, k: int = 6) -> List[Dict]:
        """
        Get most relevant chunks from pre-loaded chunks using semantic similarity.
        
        Args:
            chunks: List of chunk dictionaries
            query: Query string for semantic search
            k: Number of chunks to return
            
        Returns:
            List of most relevant chunks
        """
        if not chunks:
            return []
        
        try:
            # Create a temporary in-memory vector store for semantic search
            from langchain.docstore.document import Document
            
            docs = [Document(page_content=chunk["content"], metadata=chunk["metadata"]) 
                   for chunk in chunks]
            
            # Simple approach: use ChromaDB's similarity search if available
            # For now, return first k chunks that are substantial
            substantial_chunks = [c for c in chunks if len(c["content"]) > 100]
            
            # Return diverse chunks from different pages
            diverse_chunks = []
            seen_pages = set()
            for chunk in substantial_chunks:
                page = chunk["metadata"].get("page_number")
                if page not in seen_pages or len(diverse_chunks) < k:
                    diverse_chunks.append(chunk)
                    seen_pages.add(page)
                if len(diverse_chunks) >= k:
                    break
            
            return diverse_chunks[:k]
            
        except Exception as e:
            print(f"Error getting relevant chunks: {str(e)}")
            return chunks[:k]
    
    def _format_chat_history(self, chat_history: List[Dict], max_messages: int = 10) -> str:
        """
        Format chat history for inclusion in prompts.
        
        Args:
            chat_history: List of message dictionaries
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted history string
        """
        # Get recent messages
        recent = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history
        
        formatted = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"]
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


# Standalone testing function
def run_standalone_test():
    """Test the interactive tutor tool standalone"""
    print("Interactive Tutor Tool - Standalone Test")
    print("=" * 80)
    
    tutor = InteractiveTutorTool()
    
    # List available chapters
    print("\nAvailable chapters:")
    chapters = tutor.get_available_chapters()
    for i, ch in enumerate(chapters, 1):
        print(f"{i}. {ch['subject']} - {ch['chapter']} (Class {ch['class_level']})")
    
    if not chapters:
        print("No chapters found in database.")
        return
    
    # Use first chapter for testing
    test_chapter = chapters[0]
    print(f"\nTesting with: {test_chapter['subject']} - {test_chapter['chapter']}")
    
    # Extract topics
    print("\nExtracting topics...")
    topics = tutor.extract_topics(
        test_chapter['subject'],
        test_chapter['chapter'],
        test_chapter['class_level']
    )
    print("Topics found:")
    for topic in topics:
        print(f"  - {topic}")
    
    # Load chunks
    print("\nLoading chapter chunks...")
    chunks = tutor.load_chapter_chunks(
        test_chapter['subject'],
        test_chapter['chapter'],
        test_chapter['class_level']
    )
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate initial teaching
    print("\nGenerating initial teaching message...")
    initial = tutor.generate_initial_teaching(
        test_chapter['subject'],
        test_chapter['chapter'],
        test_chapter['class_level'],
        topics,
        chunks
    )
    print(f"\nTeacher: {initial}")


if __name__ == "__main__":
    run_standalone_test()

