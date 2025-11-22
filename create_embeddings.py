import os
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

load_dotenv()

# Use absolute path based on this file's location
# This file is in AI/, so chroma_store should be at AI/chroma_store
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")
MODEL_NAME = "BAAI/bge-small-en"
DEVICE = "cpu"
COLLECTION_NAME = "chapter_embeddings"

print(f"📁 ChromaDB directory: {CHROMA_DIR}")

def process_pdf_to_documents(file_path: str, subject: str, chapter: str, class_level: str, document_id: str = None, study_material_id: str = None):
    """
    Return (document_id, list_of_docs, list_of_ids, list_of_metadatas)
    where each id is stable: "{document_id}::{page}::{chunk}"
    
    Args:
        file_path: Path to PDF file
        subject: Subject name
        chapter: Chapter/title name
        class_level: Class level/grade
        document_id: Document ID (used for chunking, defaults to study_material_id if not provided)
        study_material_id: Study material ID from PostgreSQL (stored in metadata for filtering)
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # Use study_material_id as document_id if document_id not provided
    document_id = document_id or study_material_id or str(uuid.uuid4())
    # study_material_id defaults to document_id if not explicitly provided
    study_material_id = study_material_id or document_id

    docs = []
    ids = []
    metadatas = []

    for page_idx, page in enumerate(pages):
        page_chunks = splitter.split_text(page.page_content or "")

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{document_id}::{page_idx+1}::{chunk_idx}"
            metadata = {
                "document_id": document_id,
                "study_material_id": study_material_id,  # Add study_material_id for filtering
                "file_name": os.path.basename(file_path),
                "subject": subject,
                "chapter": chapter,
                "class_level": str(class_level),
                "page_number": page_idx + 1,
                "chunk_index": chunk_idx,                
            }
            docs.append(Document(page_content=chunk_text, metadata=metadata))
            ids.append(chunk_id)
            metadatas.append(metadata)

    return document_id, docs, ids, metadatas

def delete_existing_document(db, document_id):
    """
    If the adapter supports a delete_by_metadata or delete(ids) API, use that.
    Fallback: some Chroma clients allow db.delete_collection or manual filter deletion.
    """
    try:
        # If your chorma adapter has a delete API (pseudocode)
        db.delete(filter={"document_id": document_id})
    except Exception:
        # Best-effort: if not available, you can warn and skip;
        # or delete entire collection and re-ingest (if acceptable)
        print("delete by metadata not supported by this adapter; ensure idempotency manually")

def ingest_file_to_chroma(file_path, subject, chapter, class_level, document_id=None, study_material_id=None):
    """
    Ingest file to ChromaDB with embeddings.
    
    Args:
        file_path: Path to PDF file
        subject: Subject name
        chapter: Chapter/title name
        class_level: Class level/grade
        document_id: Document ID (used for chunking)
        study_material_id: Study material ID from PostgreSQL (stored in metadata for filtering)
    """
    print(f"📁 ChromaDB directory: {CHROMA_DIR}")
    print(f"📁 Collection name: {COLLECTION_NAME}")
    print(f"📄 Processing file: {file_path}")
    print(f"🆔 Document ID: {document_id}, Study Material ID: {study_material_id}")
    
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": DEVICE})
    db = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model, persist_directory=CHROMA_DIR)

    document_id, docs, ids, metadatas = process_pdf_to_documents(file_path, subject, chapter, class_level, document_id, study_material_id)
    
    print(f"📊 Created {len(docs)} document chunks")
    if docs:
        print(f"📋 Sample metadata: {metadatas[0]}")

    # 1. remove existing chunks for this document_id to avoid duplicates (idempotent)
    print(f"🗑️  Deleting existing chunks for document_id: {document_id}")
    delete_existing_document(db, document_id)

    # 2. attempt to add with explicit ids (preferred)
    print(f"➕ Adding {len(docs)} chunks to ChromaDB...")
    try:
        # many Chroma adapters allow add_texts(texts, metadatas, ids=ids)
        texts = [d.page_content for d in docs]
        db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"✅ Successfully added chunks using add_texts()")
    except TypeError:
        # fallback: some wrappers expect Document objects
        print(f"⚠️  add_texts() failed, trying add_documents()...")
        db.add_documents(docs)  # may not accept ids, in which case you must avoid duplicates otherwise
        print(f"✅ Successfully added chunks using add_documents()")

    # 3. Persistence is automatic when using persist_directory in Chroma constructor
    # No need to call persist() - ChromaDB automatically persists when persist_directory is set
    print(f"💾 ChromaDB will auto-persist to: {CHROMA_DIR}")
    print(f"✅ Embeddings saved (persistence is automatic with persist_directory)")

    print(f"✅ Ingested {len(docs)} chunks for document {document_id} (study_material_id: {study_material_id})")
    return document_id

# Example usage
if __name__ == "__main__":
    ingest_file_to_chroma(
        file_path="../A Brief History of India_text.pdf",  # PDF is in parent directory
        subject="History",
        chapter="A Brief History of India",
        class_level="10",
        document_id="history_india_001"  # pass existing id to re-ingest/update
    )
