"""Document ingestion script to chunk, embed, and index sample files into ChromaDB."""

import os
import sys
import glob
import logging
from typing import List, Dict, Any

# Add src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from selfhealing_rag.config import settings
from selfhealing_rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simple_markdown_chunker(file_path: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
    """Simple paragraph and section-aware chunker for markdown/text files."""
    filename = os.path.basename(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by double newline (paragraphs/sections)
    raw_sections = text.split("\n\n")
    chunks = []
    
    current_chunk = ""
    chunk_index = 1
    
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
            
        if len(current_chunk) + len(section) <= chunk_size:
            current_chunk += ("\n\n" + section if current_chunk else section)
        else:
            if current_chunk:
                chunks.append({
                    "id": f"{filename}_chunk_{chunk_index}",
                    "content": current_chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_index": chunk_index,
                        "file_path": file_path
                    }
                })
                chunk_index += 1
            current_chunk = section

    if current_chunk:
        chunks.append({
            "id": f"{filename}_chunk_{chunk_index}",
            "content": current_chunk,
            "metadata": {
                "source": filename,
                "chunk_index": chunk_index,
                "file_path": file_path
            }
        })

    return chunks


def ingest_sample_docs(docs_dir: str = "./data/sample_docs"):
    """Find all markdown/text files in docs_dir, chunk them, and index in ChromaDB."""
    logger.info(f"Starting ingestion from directory: {docs_dir}")
    
    files = glob.glob(os.path.join(docs_dir, "*.md")) + glob.glob(os.path.join(docs_dir, "*.txt"))
    if not files:
        logger.warning(f"No markdown or text files found in {docs_dir}")
        return

    vector_store = ChromaVectorStore()

    all_ids = []
    all_docs = []
    all_metadatas = []

    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        chunks = simple_markdown_chunker(file_path)
        for c in chunks:
            all_ids.append(c["id"])
            all_docs.append(c["content"])
            all_metadatas.append(c["metadata"])

    logger.info(f"Upserting {len(all_docs)} total chunks into ChromaDB...")
    vector_store.add_documents(ids=all_ids, documents=all_docs, metadatas=all_metadatas)
    logger.info("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_sample_docs()
