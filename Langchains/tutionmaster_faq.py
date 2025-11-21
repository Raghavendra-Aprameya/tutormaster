import json
import os
import time
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import AzureChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Chroma configuration (matching create_embeddings.py)
# tutionmaster_faq.py is in AI/Langchains/, so we need ../../chroma_store to reach the root
CHROMA_DIR = "../../chroma_store"
MODEL_NAME = "BAAI/bge-small-en"
DEVICE = "cpu"
COLLECTION_NAME = "chapter_embeddings"
api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class TutorLLMNode:
    """
    LLM Node that queries ChromaDB and generates responses using Azure OpenAI
    """
    
    def __init__(self):
        """Initialize the embedding model and ChromaDB connection"""
        # Initialize embedding model (same as used during ingestion)
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
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=openai_endpoint,
            api_version="2024-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.4,
            
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are an educational tutor assistant teaching from a PDF textbook (A Brief History of India - Class 10).

CRITICAL: Answer ONLY from the provided PDF content. NEVER use general knowledge.

PDF Content Retrieved:
{context}

Student's Question: {query}

Instructions: 
- Provide a factual response based STRICTLY on the PDF content above
- Use ONLY information present in the retrieved excerpts
- If the information is not in the PDF excerpts, say "This specific information is not covered in our textbook"
- Do NOT add information from general knowledge or assumptions
- Focus on facts and details explicitly stated in the PDF content

Educational Response (from PDF only):"""
        )
    
    def query_and_respond(self, query_text: str, filter_meta: dict = None, k: int = 6):
        """
        Query the ChromaDB and generate a response using LLM
        
        Args:
            query_text (str): The user's query
            filter_meta (dict): Optional ChromaDB filter with proper operators
                Examples:
                - Single condition: {"subject": {"$eq": "Geography"}}
                - Multiple conditions: {"$and": [{"subject": {"$eq": "History"}}, {"class_level": {"$eq": "10"}}]}
            k (int): Number of chunks to retrieve (default: 6, higher values help get content beyond just titles)
        
        Returns:
            dict: Contains the response, retrieved chunks, and metadata
        """
        try:
            # Query ChromaDB for similar chunks
            print(f"\n🔍 Querying ChromaDB for: '{query_text}'")
            if filter_meta:
                print(f"   Filters: {filter_meta}")
            
            results = self.db.similarity_search(
                query_text,
                k=k,
                filter=filter_meta
            )
            
            print(f"\n📚 Retrieved {len(results)} chunks:")
            
            # Extract and display retrieved chunks
            retrieved_chunks = []
            for idx, doc in enumerate(results, 1):
                chunk_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                retrieved_chunks.append(chunk_info)
                
                print(f"\n--- Chunk {idx} ---")
                print(f"Metadata: {doc.metadata}")
                print(f"Content ({len(doc.page_content)} chars): {doc.page_content[:400]}...")
            
            # Prepare context for LLM
            context = "\n\n".join([
                f"[Source {i+1} - {chunk['metadata'].get('subject', 'N/A')}, "
                f"Chapter: {chunk['metadata'].get('chapter', 'N/A')}, "
                f"Page: {chunk['metadata'].get('page_number', 'N/A')}]\n{chunk['content']}"
                for i, chunk in enumerate(retrieved_chunks)
            ])
            
            # Debug: Show full context being sent to LLM
            print(f"\n📝 Context being sent to LLM ({len(context)} characters):")
            print("=" * 80)
            print(context[:1500] + "..." if len(context) > 1500 else context)
            print("=" * 80)
            
            # Generate response using LLM
            print(f"\n🤖 Generating response with LLM...")
            prompt = self.prompt_template.format(query=query_text, context=context)
            response = self.llm.invoke(prompt)
            
            # Extract response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            print(f"\n✅ Response generated successfully!")
            
            return {
                "query": query_text,
                "response": response_text,
                "retrieved_chunks": retrieved_chunks,
                "num_chunks": len(retrieved_chunks),
                "filters_applied": filter_meta
            }
            
        except Exception as e:
            print(f"\n❌ Error during query and response: {str(e)}")
            return {
                "query": query_text,
                "response": f"Error: {str(e)}",
                "retrieved_chunks": [],
                "num_chunks": 0,
                "filters_applied": filter_meta,
                "error": str(e)
            }
    
    def query_and_respond_with_history(self, query_text: str, chat_history: list = None, 
                                       filter_meta: dict = None, k: int = 6):
        """
        Query ChromaDB and generate a response using conversation history for context.
        
        Args:
            query_text (str): The user's query
            chat_history (list): Optional conversation history as list of dicts with 'role' and 'content'
            filter_meta (dict): Optional ChromaDB filter with proper operators
            k (int): Number of chunks to retrieve (default: 6)
        
        Returns:
            dict: Contains the response, retrieved chunks, and metadata
        """
        try:
            # Query ChromaDB for similar chunks
            results = self.db.similarity_search(
                query_text,
                k=k,
                filter=filter_meta
            )
            
            # Extract retrieved chunks
            retrieved_chunks = []
            for doc in results:
                chunk_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                retrieved_chunks.append(chunk_info)
            
            # Prepare context for LLM
            context = "\n\n".join([
                f"[Source {i+1} - {chunk['metadata'].get('subject', 'N/A')}, "
                f"Chapter: {chunk['metadata'].get('chapter', 'N/A')}, "
                f"Page: {chunk['metadata'].get('page_number', 'N/A')}]\n{chunk['content']}"
                for i, chunk in enumerate(retrieved_chunks)
            ])
            
            # Build prompt with chat history if provided
            if chat_history and len(chat_history) > 0:
                # Format recent conversation history
                history_text = self._format_history_for_prompt(chat_history[-6:])  # Last 6 messages
                
                # Enhanced prompt with history - strict PDF-only
                prompt_with_history = f"""You are teaching from a PDF textbook (A Brief History of India - Class 10).

CRITICAL: Answer ONLY from the PDF content provided below. NEVER use general knowledge.

Conversation History:
{history_text}

PDF Content Retrieved:
{context}

Current Question: {query_text}

Instructions:
- Provide answer based STRICTLY on the PDF content above
- Use ONLY information present in the retrieved PDF excerpts
- If information is not in the excerpts, say "This is not covered in our textbook"
- Do NOT add information from general knowledge
- If conversation history is relevant, acknowledge it naturally

Answer (from PDF only):"""
                
                response = self.llm.invoke(prompt_with_history)
            else:
                # Use original prompt without history
                prompt = self.prompt_template.format(query=query_text, context=context)
                response = self.llm.invoke(prompt)
            
            # Extract response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "query": query_text,
                "response": response_text,
                "retrieved_chunks": retrieved_chunks,
                "num_chunks": len(retrieved_chunks),
                "filters_applied": filter_meta
            }
            
        except Exception as e:
            return {
                "query": query_text,
                "response": f"Error: {str(e)}",
                "retrieved_chunks": [],
                "num_chunks": 0,
                "filters_applied": filter_meta,
                "error": str(e)
            }
    
    def _format_history_for_prompt(self, chat_history: list) -> str:
        """
        Format chat history for inclusion in prompts.
        
        Args:
            chat_history: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted history string
        """
        formatted = []
        for msg in chat_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


# Example usage
if __name__ == "__main__":
    # Initialize the tutor node
    tutor_node = TutorLLMNode()
    
    # Example 1: Query with multiple filters (using $and operator)
    print("=" * 80)
    print("EXAMPLE 1: Query with multiple filters")
    print("=" * 80)
    
    query_text = "What happened when Mahmud of Ghazni attacked the Somnath temple?"
    # ChromaDB filter format with $and for multiple conditions
    filter_meta = {
        "$and": [
            {"subject": {"$eq": "History"}},
            {"class_level": {"$eq": "10"}}
        ]
    }
    
    result = tutor_node.query_and_respond(
        query_text=query_text,
        filter_meta=filter_meta,
        k=6  # Increased to 6 to get more context chunks beyond just titles
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESPONSE TO USER:")
    print("=" * 80)
    print(result["response"])
    print("\n")
    