"""
Revision Pointers Generator
Generates a concise list of last-minute revision pointers from the entire chapter content.
Designed for quick exam preparation and review.
"""

import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Use absolute path based on this file's location
# This file is in AI/Langchains/, so chroma_store should be at AI/chroma_store
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")
MODEL_NAME = "BAAI/bge-small-en"
DEVICE = "cpu"
COLLECTION_NAME = "chapter_embeddings"

print(f"📁 ChromaDB directory: {CHROMA_DIR}")


class RevisionPointersGenerator:
    """
    Generates last-minute revision pointers from chapter content.
    """
    
    def __init__(self, subject: str = None, class_level: str = None, chapter: str = None, study_material_id: str = None):
        """
        Initialize the revision pointers generator
        
        Args:
            subject: Subject name (e.g., "History")
            class_level: Class level/grade (e.g., "10")
            chapter: Chapter/title name (e.g., "A Brief History of India")
            study_material_id: Study material ID from PostgreSQL (used to filter embeddings)
        """
        self.subject = subject or "History"
        self.class_level = class_level or "10"
        self.chapter = chapter or "A Brief History of India"
        self.study_material_id = study_material_id
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": DEVICE}
        )
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_DIR
        )
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.5,  # Balanced temperature for structured but creative output
        )
    
    def _get_chapter_content(self, k: int = 50) -> str:
        """
        Retrieve comprehensive chapter content from ChromaDB.
        
        Args:
            k: Number of documents to retrieve (increased for comprehensive coverage)
        
        Returns:
            Combined chapter content as a string
        """
        # Build filter based on available parameters
        filter_conditions = []
        
        # If study_material_id is provided, filter by it (primary filter)
        # Try both study_material_id and document_id (for backward compatibility)
        if self.study_material_id:
            # First try filtering by study_material_id
            filter_conditions.append({"study_material_id": {"$eq": str(self.study_material_id)}})
            print(f"🔍 Filtering by study_material_id: {self.study_material_id}")
        
        # Add subject, chapter, and class_level filters if provided
        if self.subject:
            filter_conditions.append({"subject": {"$eq": self.subject}})
        if self.chapter:
            filter_conditions.append({"chapter": {"$eq": self.chapter}})
        if self.class_level:
            filter_conditions.append({"class_level": {"$eq": str(self.class_level)}})
        
        # Build filter metadata
        if filter_conditions:
            if len(filter_conditions) == 1:
                filter_meta = filter_conditions[0]
            else:
                filter_meta = {"$and": filter_conditions}
        else:
            filter_meta = None
        
        # Use a broad query to get comprehensive content
        query_text = f"complete summary of {self.chapter or 'the document'} all topics important points"
        
        print(f"🔍 Querying ChromaDB with filter: {filter_meta}")
        print(f"🔍 Query text: {query_text}")
        print(f"🔍 Requesting {k} chunks")
        
        if filter_meta:
            results = self.db.similarity_search(
                query_text,
                k=k,
                filter=filter_meta
            )
        else:
            results = self.db.similarity_search(query_text, k=k)
        
        if not results:
            print(f"⚠️ Warning: No documents found with filter: {filter_meta}")
            # Try fallback: filter by document_id if study_material_id didn't work
            if self.study_material_id:
                print(f"🔄 Trying fallback: filtering by document_id instead of study_material_id")
                fallback_conditions = [{"document_id": {"$eq": str(self.study_material_id)}}]
                if self.subject:
                    fallback_conditions.append({"subject": {"$eq": self.subject}})
                if self.chapter:
                    fallback_conditions.append({"chapter": {"$eq": self.chapter}})
                if self.class_level:
                    fallback_conditions.append({"class_level": {"$eq": str(self.class_level)}})
                
                if len(fallback_conditions) == 1:
                    fallback_filter = fallback_conditions[0]
                else:
                    fallback_filter = {"$and": fallback_conditions}
                
                print(f"🔍 Fallback filter: {fallback_filter}")
                results = self.db.similarity_search(query_text, k=k, filter=fallback_filter)
                if results:
                    print(f"✅ Fallback successful: Found {len(results)} chunks using document_id")
                else:
                    print(f"❌ Fallback also returned 0 results")
            
            if not results:
                return ""
        
        print(f"✅ Retrieved {len(results)} document chunks from ChromaDB")
        # Print sample metadata to debug
        if results:
            sample_metadata = results[0].metadata
            print(f"📋 Sample chunk metadata: {sample_metadata}")
        
        return "\n\n".join([doc.page_content for doc in results])
        
        print(f"✅ Retrieved {len(results)} document chunks from ChromaDB")
        # Print sample metadata to debug
        if results:
            sample_metadata = results[0].metadata
            print(f"📋 Sample chunk metadata: {sample_metadata}")
        
        return "\n\n".join([doc.page_content for doc in results])
    
    def _generate_revision_pointers(self, content: str) -> List[str]:
        """
        Generate concise revision pointers from chapter content.
        
        Args:
            content: The chapter content to analyze
        
        Returns:
            List of revision pointers
        """
        prompt_template = PromptTemplate(
            input_variables=["content", "subject", "class_level", "chapter"],
            template="""You are an expert educator preparing last-minute revision pointers for a {subject} exam (Class {class_level}).

Chapter: {chapter}

Textbook Content:
{content}

TASK: Create a comprehensive list of concise, exam-focused revision pointers that cover all important topics from this chapter.

REQUIREMENTS:
1. Generate 20-30 key revision pointers that cover the entire chapter
2. Each pointer should be:
   - Concise (1-2 sentences maximum)
   - Exam-focused (highlight facts, dates, names, concepts likely to appear in exams)
   - Easy to remember and review quickly
   - Organized by topic/theme where possible
3. Focus on:
   - Important dates, events, and historical periods
   - Key personalities and their contributions
   - Major concepts, movements, and their significance
   - Cause-effect relationships
   - Important facts and figures
4. Prioritize information that is commonly tested in exams

OUTPUT FORMAT:
Return ONLY a JSON array of strings. Each string is one revision pointer.
Do NOT include any additional text, explanations, or formatting outside the JSON array.

Example format:
[
    "The Maurya Empire was founded by Chandragupta Maurya around 322 BCE and was one of the largest empires in ancient India.",
    "Ashoka the Great converted to Buddhism after the Kalinga War and spread Buddhist teachings across Asia.",
    "The Gupta Period (320-550 CE) is known as the Golden Age of India due to achievements in science, mathematics, and arts."
]

Generate the revision pointers now:"""
        )
        
        full_prompt = prompt_template.format(
            content=content,
            subject=self.subject,
            class_level=self.class_level,
            chapter=self.chapter
        )
        
        response = self.llm.invoke(full_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # Extract JSON array from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                pointers = json.loads(json_str)
                
                # Validate that it's a list of strings
                if isinstance(pointers, list):
                    # Filter out any non-string items and ensure all are strings
                    pointers = [str(p).strip() for p in pointers if p]
                    return pointers
                else:
                    print(f"Warning: Expected list but got {type(pointers)}")
                    return []
            else:
                # Fallback: try to parse as newline-separated list
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                # Filter out lines that look like JSON structure markers
                lines = [line for line in lines if not line.startswith('[') and not line.startswith(']')]
                # Remove numbering if present (e.g., "1. ", "- ", "* ")
                cleaned_lines = []
                for line in lines:
                    # Remove common list markers
                    for marker in ['1.', '2.', '3.', '4.', '5.', '- ', '* ', '• ']:
                        if line.startswith(marker):
                            line = line[len(marker):].strip()
                            break
                    if line and not line.startswith('{') and not line.startswith('"') and ':' not in line[:20]:
                        cleaned_lines.append(line)
                return cleaned_lines[:30]  # Limit to 30 pointers
                
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM: {str(e)}")
            print(f"Response: {response_text[:500]}")
            # Fallback: try to extract pointers from text
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            cleaned_lines = []
            for line in lines:
                # Skip JSON structure and metadata
                if line.startswith('[') or line.startswith(']') or line.startswith('{'):
                    continue
                # Remove list markers
                for marker in ['1.', '2.', '3.', '- ', '* ', '• ']:
                    if line.startswith(marker):
                        line = line[len(marker):].strip()
                        break
                # Remove quotes if present
                line = line.strip('"').strip("'").strip()
                if line and len(line) > 10:  # Only include substantial pointers
                    cleaned_lines.append(line)
            return cleaned_lines[:30]
        except Exception as e:
            print(f"Error generating revision pointers: {str(e)}")
            return []
    
    def generate_revision_pointers(self) -> Dict[str, Any]:
        """
        Generate revision pointers for the chapter.
        
        Returns:
            Dictionary containing subject, class_level, chapter, and list of pointers
        """
        content = self._get_chapter_content(k=50)
        if not content:
            print(f"Warning: No content found for study_material_id: {self.study_material_id}")
            return {
                "subject": self.subject,
                "class_level": self.class_level,
                "chapter": self.chapter,
                "pointers": [],
                "total_pointers": 0
            }
        
        pointers = self._generate_revision_pointers(content)
        
        return {
            "subject": self.subject,
            "class_level": self.class_level,
            "chapter": self.chapter,
            "pointers": pointers,
            "total_pointers": len(pointers)
        }


def main():
    """Test the revision pointers generator"""
    generator = RevisionPointersGenerator()
    result = generator.generate_revision_pointers()
    
    print("=" * 80)
    print("REVISION POINTERS GENERATOR TEST")
    print("=" * 80)
    print(f"\nSubject: {result['subject']}")
    print(f"Class Level: {result['class_level']}")
    print(f"Chapter: {result['chapter']}")
    print(f"\nTotal Pointers: {result['total_pointers']}")
    print("\n" + "=" * 80)
    print("REVISION POINTERS:")
    print("=" * 80)
    
    for i, pointer in enumerate(result['pointers'], 1):
        print(f"\n{i}. {pointer}")
    
    print("\n" + "=" * 80)
    
    # Save to file
    output_file = "revision_pointers.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()

