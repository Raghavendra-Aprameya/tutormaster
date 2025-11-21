import os
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "../chroma_store"
MODEL_NAME = "BAAI/bge-small-en"
DEVICE = "cpu"
COLLECTION_NAME = "chapter_embeddings"

def process_pdf_to_documents(file_path: str, subject: str, chapter: str, class_level: str, document_id: str = None):
    """
    Return (document_id, list_of_docs, list_of_ids, list_of_metadatas)
    where each id is stable: "{document_id}::{page}::{chunk}"
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_id = document_id or str(uuid.uuid4())

    docs = []
    ids = []
    metadatas = []

    for page_idx, page in enumerate(pages):
        page_chunks = splitter.split_text(page.page_content or "")

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{document_id}::{page_idx+1}::{chunk_idx}"
            metadata = {
                "document_id": document_id,
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

def ingest_file_to_chroma(file_path, subject, chapter, class_level, document_id=None):
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": DEVICE})
    db = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model, persist_directory=CHROMA_DIR)

    document_id, docs, ids, metadatas = process_pdf_to_documents(file_path, subject, chapter, class_level, document_id)

    # 1. remove existing chunks for this document_id to avoid duplicates (idempotent)
    delete_existing_document(db, document_id)

    # 2. attempt to add with explicit ids (preferred)
    try:
        # many Chroma adapters allow add_texts(texts, metadatas, ids=ids)
        texts = [d.page_content for d in docs]
        db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    except TypeError:
        # fallback: some wrappers expect Document objects
        db.add_documents(docs)  # may not accept ids, in which case you must avoid duplicates otherwise

    # 3. persist
    try:
        db.persist()
    except Exception as e:
        print("persist failed:", e)

    print(f"Ingested {len(docs)} chunks for document {document_id}")
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
