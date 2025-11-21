"""
Exam Question Generator Node
Generates important exam-style questions and answers from the ChromaDB documents.
Hardcoded to: Subject=History, Class Level=10, Chapter="A Brief History of India"
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Hardcoded configuration
SUBJECT = "History"
CLASS_LEVEL = "10"
CHAPTER = "A Brief History of India"
CHROMA_DIR = "../../chroma_store"
MODEL_NAME = "BAAI/bge-small-en"
DEVICE = "cpu"
COLLECTION_NAME = "chapter_embeddings"

# ChromaDB filter
FILTER_META = {
    "$and": [
        {"subject": {"$eq": SUBJECT}},
        {"class_level": {"$eq": CLASS_LEVEL}}
    ]
}


class ExamQuestionGenerator:
    """
    Generates exam-style questions and answers from ChromaDB documents.
    """
    
    def __init__(self):
        """Initialize the exam question generator"""
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
            temperature=0.7,
        )
    
    def retrieve_document_chunks(self, k: int = 30) -> List[Dict]:
        """
        Retrieve relevant chunks from ChromaDB for question generation.
        Uses a general query to get diverse content from the document.
        
        Args:
            k: Number of chunks to retrieve (default: 30 for comprehensive coverage)
            
        Returns:
            List of document chunks with content and metadata
        """
        # Use a general query to get diverse content
        query = f"{CHAPTER} {SUBJECT} Class {CLASS_LEVEL}"
        
        print(f"Retrieving {k} chunks from ChromaDB...")
        results = self.db.similarity_search(
            query,
            k=k,
            filter=FILTER_META
        )
        
        chunks = []
        for doc in results:
            chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        print(f"Retrieved {len(chunks)} chunks")
        return chunks
    
    def generate_exam_questions(self, num_questions: int = 10) -> List[Dict]:
        """
        Generate exam-style questions and answers from document content.
        
        Args:
            num_questions: Number of questions to generate (default: 10)
            
        Returns:
            List of dictionaries containing question, answer, and metadata
        """
        # Retrieve document chunks
        chunks = self.retrieve_document_chunks(k=30)
        
        if not chunks:
            return []
        
        # Prepare context from chunks
        context = "\n\n".join([
            f"[Source {i+1} - Page {chunk['metadata'].get('page_number', 'N/A')}]\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Create prompt for question generation
        prompt_template = PromptTemplate(
            input_variables=["context", "num_questions", "subject", "class_level", "chapter"],
            template="""You are an expert educator creating exam questions for {subject} Class {class_level}.

You have access to content from the textbook chapter: "{chapter}"

PDF Content:
{context}

TASK: Generate exactly {num_questions} important exam-style questions and answers based on the PDF content above.

REQUIREMENTS:
1. Questions should be important from an exam perspective (focus on key facts, dates, events, people, concepts)
2. Questions should cover different topics from the chapter
3. Mix different question types:
   - Factual questions (What, Who, When, Where)
   - Conceptual questions (Why, How, Explain)
   - Analytical questions (Compare, Describe, Discuss)
4. Answers must be based STRICTLY on the PDF content provided
5. Answers should be concise but complete (2-4 sentences typically)
6. Include important dates, names, and facts
7. Questions should be suitable for Class {class_level} level

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "Question text here?",
    "answer": "Complete answer based on PDF content.",
    "topic": "Topic/subject area",
    "difficulty": "easy|medium|hard",
    "question_type": "factual|conceptual|analytical"
  }},
  ...
]

Return ONLY the JSON array, no additional text or explanation.

Questions and Answers:"""
        )
        
        prompt = prompt_template.format(
            context=context[:15000],  # Limit context to avoid token limits
            num_questions=num_questions,
            subject=SUBJECT,
            class_level=CLASS_LEVEL,
            chapter=CHAPTER
        )
        
        print(f"Generating {num_questions} exam questions...")
        
        # Generate questions using LLM
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response
        try:
            # Try to find JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                questions = json.loads(json_str)
                
                # Validate and clean questions
                validated_questions = []
                for q in questions:
                    if isinstance(q, dict) and 'question' in q and 'answer' in q:
                        validated_questions.append({
                            "question": q.get("question", ""),
                            "answer": q.get("answer", ""),
                            "topic": q.get("topic", "General"),
                            "difficulty": q.get("difficulty", "medium"),
                            "question_type": q.get("question_type", "factual")
                        })
                
                print(f"Successfully generated {len(validated_questions)} questions")
                return validated_questions[:num_questions]  # Ensure we don't exceed requested number
            else:
                print("Error: Could not find JSON array in response")
                return []
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            print(f"Response text: {response_text[:500]}")
            return []
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []
    
    def generate_and_save(self, num_questions: int = 10, output_file: str = None) -> Dict:
        """
        Generate exam questions and optionally save to file.
        
        Args:
            num_questions: Number of questions to generate
            output_file: Optional file path to save JSON output
            
        Returns:
            Dictionary with questions and metadata
        """
        questions = self.generate_exam_questions(num_questions)
        
        result = {
            "subject": SUBJECT,
            "class_level": CLASS_LEVEL,
            "chapter": CHAPTER,
            "total_questions": len(questions),
            "questions": questions
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Questions saved to {output_file}")
        
        return result


def main():
    """Main function to generate and display exam questions"""
    print("=" * 80)
    print("EXAM QUESTION GENERATOR")
    print("=" * 80)
    print(f"Subject: {SUBJECT}")
    print(f"Class Level: {CLASS_LEVEL}")
    print(f"Chapter: {CHAPTER}")
    print("=" * 80)
    print()
    
    # Initialize generator
    generator = ExamQuestionGenerator()
    
    # Generate questions
    result = generator.generate_and_save(
        num_questions=10,
        output_file="exam_questions.json"
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("GENERATED EXAM QUESTIONS")
    print("=" * 80)
    
    for i, q in enumerate(result["questions"], 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] [{q['question_type']}] - {q['topic']}")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")
    
    print("\n" + "=" * 80)
    print(f"Total questions generated: {result['total_questions']}")
    print("=" * 80)
    
    # Also print as JSON
    print("\nJSON Output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

