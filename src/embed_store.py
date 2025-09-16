"""
Embedding store module for RAG system
Handles document embedding, FAISS indexing, and similarity search
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    torch.set_num_threads(4)  # tweak 2–8 based on your CPU cores
except Exception as _:
    pass

# Fix the import issues by using proper versions
try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Please upgrade sentence-transformers: pip install --upgrade sentence-transformers"
    )

from ingest import discover_and_chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = DEFAULT_MODEL  # For backward compatibility
ARTIFACTS_DIR = Path("artifacts")
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.jsonl"
CONFIG_FILE = "store_config.json"


class EmbeddingStore:
    """Manages embeddings and FAISS index for document retrieval"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, artifacts_dir: Path = ARTIFACTS_DIR):
        self.model_name = model_name
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.index = None
        self.metadata = []
        
        # Create artifacts directory if doesn't exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Trying with default model instead...")
                self.model = SentenceTransformer(DEFAULT_MODEL)
        return self.model
    
    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def build_index(self, chunks: List[Dict], batch_size: int = 32) -> Tuple[faiss.Index, np.ndarray]:
        """Build FAISS index from document chunks"""
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        model = self._load_model()
        
        # Extract texts
        texts = [chunk.get("text", "") for chunk in chunks]
        logger.info(f"Encoding {len(texts)} text chunks...")
        
        # Generate embeddings with progress tracking
        try:
            embeddings = model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            # Fallback to smaller batch size
            logger.info("Retrying with smaller batch size...")
            embeddings = model.encode(
                texts,
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        # Normalize for cosine similarity
        embeddings = self.normalize_vectors(embeddings.astype("float32"))
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index with dimension: {dimension}")
        
        # Using Inner Product for normalized vectors (equivalent to cosine)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        logger.info(f"Index built successfully with {index.ntotal} vectors")
        
        self.index = index
        self.metadata = chunks
        
        return index, embeddings
    
    def save(self, index: faiss.Index = None, chunks: List[Dict] = None, 
             embeddings: np.ndarray = None) -> None:
        """Save index and metadata to disk"""
        
        # Use instance variables if not provided
        index = index or self.index
        chunks = chunks or self.metadata
        
        if index is None or not chunks:
            raise ValueError("No index or metadata to save")
        
        # Save FAISS index
        index_path = self.artifacts_dir / INDEX_FILE
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        meta_path = self.artifacts_dir / METADATA_FILE
        logger.info(f"Saving metadata to {meta_path}")
        with open(meta_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "dimension": int(index.d),
            "total_vectors": int(index.ntotal),
            "index_type": "IndexFlatIP"
        }
        
        config_path = self.artifacts_dir / CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        logger.info("Artifacts saved successfully")
    
    def load(self) -> Tuple[faiss.Index, List[Dict]]:
        """Load index and metadata from disk"""
        index_path = self.artifacts_dir / INDEX_FILE
        meta_path = self.artifacts_dir / METADATA_FILE
        
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {self.artifacts_dir}. "
                "Please run indexing first."
            )
        
        # Load FAISS index
        logger.info(f"Loading index from {index_path}")
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        
        logger.info(f"Loaded {index.ntotal} vectors and {len(metadata)} metadata entries")
        
        self.index = index
        self.metadata = metadata
        
        return index, metadata
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search the FAISS index and return top_k results with metadata.
        Instance method version.
        """
        if self.index is None or not self.metadata:
            # Try to load if not loaded
            try:
                self.load()
            except FileNotFoundError:
                raise FileNotFoundError("FAISS index or metadata not found. Please run indexing first.")
        
        # Load model and encode query
        model = self._load_model()
        query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
        query_vector = self.normalize_vectors(query_vector)
        
        # Clamp top_k
        k = min(max(1, int(top_k)), len(self.metadata))
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                "id": meta.get("id", f"chunk_{idx}"),
                "title": meta.get("title", "Untitled"),
                "source": meta.get("source", "unknown"),
                "text": meta.get("text", ""),  # Include full text for context
                "text_preview": (meta.get("text", "")[:200].replace("\n", " ")).strip()
            })
        return results


def search_standalone(index: faiss.Index, meta: List[Dict], query: str, 
                     top_k: int = 5, model_name: str = DEFAULT_MODEL) -> List[Dict]:
    """
    Standalone search function to avoid class/kwarg conflicts.
    This is what rag_chat.py calls.
    
    Args:
        index: FAISS index object
        meta: List of metadata dictionaries
        query: Search query string
        top_k: Number of results to return
        model_name: Name of the embedding model to use
        
    Returns:
        List of search results with scores and metadata
    """
    # Safety check
    if index is None or not meta:
        raise ValueError("FAISS index or metadata not provided.")
    
    # Load model (cached by sentence-transformers)
    logger.info(f"Loading model for search: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        logger.info(f"Falling back to default model: {DEFAULT_MODEL}")
        model = SentenceTransformer(DEFAULT_MODEL)
    
    # Encode and normalize (cosine)
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_vector = query_vector / (norms + 1e-10)
    
    # Clamp top_k to available results
    k = min(max(1, int(top_k)), len(meta))
    
    # Search the index
    scores, indices = index.search(query_vector, k)
    
    # Prepare results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        results.append({
            "score": float(score),
            "id": m.get("id", f"chunk_{idx}"),
            "title": m.get("title", "Untitled"),
            "source": m.get("source", "unknown"),
            "text": m.get("text", ""),  # Include full text
            "text_preview": (m.get("text", "")[:200].replace("\n", " ")).strip()
        })
    
    return results


# Global store instance for backward compatibility
_global_store = EmbeddingStore()


# Backward compatibility functions
def load_index_and_meta() -> Tuple[faiss.Index, List[Dict]]:
    """Load index and metadata - backward compatibility function"""
    logger.info("[embed_store] load_index_and_meta called")
    return _global_store.load()


def search(query: str, top_k: int = 5) -> List[Dict]:
    """Search function - backward compatibility (when called with just query and top_k)"""
    logger.info(f"[embed_store] search called with query: {query[:50]}...")
    return _global_store.search(query, top_k)


def build_index(chunks: List[Dict], batch_size: int = 32) -> Tuple[faiss.Index, np.ndarray]:
    """Build index - backward compatibility function"""
    logger.info("[embed_store] build_index called")
    return _global_store.build_index(chunks, batch_size)


def save_artifacts(index: faiss.Index, chunks: List[Dict], 
                  embeddings: np.ndarray, model_name: str) -> None:
    """Save artifacts - backward compatibility function"""
    logger.info("[embed_store] save_artifacts called")
    _global_store.model_name = model_name
    _global_store.save(index, chunks, embeddings)


def rebuild_from_data(data_dir: str = "data") -> bool:
    """Rebuild index from data directory"""
    logger.info(f"[embed_store] rebuild_from_data called for directory: {data_dir}")
    
    # Discover and chunk documents
    chunks = discover_and_chunk(data_dir)
    if not chunks:
        raise RuntimeError(f"No chunks found in {data_dir} while rebuilding.")
    
    logger.info(f"Found {len(chunks)} chunks to index")
    
    # Build index and save
    idx, embs = _global_store.build_index(chunks)
    _global_store.save(idx, chunks, embs)
    
    logger.info("Index rebuilt successfully")
    return True


def get_store_info() -> Dict:
    """Get information about the current store"""
    config_path = ARTIFACTS_DIR / CONFIG_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {"status": "No index found"}


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and manage embedding store")
    parser.add_argument("--data-dir", default="data", help="Directory containing documents")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--search", type=str, help="Test search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--info", action="store_true", help="Show store information")
    
    args = parser.parse_args()
    
    # Show info if requested
    if args.info:
        info = get_store_info()
        print("\nEmbedding Store Information:")
        print("-" * 40)
        for key, value in info.items():
            print(f"{key}: {value}")
        return
    
    # Initialize store
    store = EmbeddingStore(model_name=args.model)
    
    # Handle search
    if args.search and not args.rebuild:
        # Just search if index exists
        try:
            store.load()  # Load the index first
            results = store.search(args.search, top_k=args.top_k)
            print(f"\nSearch results for: '{args.search}'\n")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Source: {result['source']}")
                print(f"   Title: {result['title']}")
                print(f"   Preview: {result['text_preview']}\n")
        except FileNotFoundError:
            print("Index not found. Building index first...")
            args.rebuild = True
    
    # Handle rebuild or initial build
    if args.rebuild or not (store.artifacts_dir / INDEX_FILE).exists():
        # Build index
        logger.info(f"Discovering documents in {args.data_dir}")
        chunks = discover_and_chunk(args.data_dir)
        
        if not chunks:
            logger.error(f"No documents found in {args.data_dir}")
            return
        
        logger.info(f"Found {len(chunks)} chunks")
        
        # Build and save
        index, embeddings = store.build_index(chunks)
        store.save(index, chunks, embeddings)
        
        print(f"\nIndex built successfully!")
        print(f"Total chunks indexed: {len(chunks)}")
        print(f"Index dimension: {index.d}")
        print(f"Files saved to: {store.artifacts_dir}")
        
        # Test search if query provided
        if args.search:
            print("\n" + "="*60)
            store.load()  # Reload to test
            results = store.search(args.search, top_k=args.top_k)
            print(f"\nSearch results for: '{args.search}'\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Source: {result['source']}")
                print(f"   Title: {result['title']}")
                print(f"   Preview: {result['text_preview']}\n")


if __name__ == "__main__":
    main()