import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict
import time

CHUNKS_FILE = "data/processed_chunks.json"
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "student_mentor_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingPipeline:
    
    def __init__(self):
        self.model = None
        self.client = None
        self.collection = None
        
    def load_embedding_model(self):
        print("\n🔄 Loading embedding model...")
        print(f"   Model: {EMBEDDING_MODEL}")
        start_time = time.time()
        
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.2f} seconds")
        
    def load_chunks(self) -> List[Dict]:
        print(f"\n📂 Loading chunks from {CHUNKS_FILE}...")
        
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"   ✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def initialize_chromadb(self):
        print(f"\n🗄️  Initializing ChromaDB...")
        print(f"   Directory: {CHROMA_DB_DIR}")
        
        Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            print(f"   ♻️  Deleted existing collection")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"   ✅ Collection '{COLLECTION_NAME}' created")
    
    def generate_and_store_embeddings(self, chunks: List[Dict]):
        print(f"\n🧮 Generating embeddings for {len(chunks)} chunks...")
        print("   This may take a few minutes...")
        
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_index': chunk['chunk_index'],
                'char_count': chunk['char_count']
            }
            for chunk in chunks
        ]
        
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        embed_time = time.time() - start_time
        print(f"\n   ✅ Generated {len(embeddings)} embeddings in {embed_time:.2f} seconds")
        print(f"   ⚡ Speed: {len(embeddings)/embed_time:.1f} chunks/second")
        
        print("\n💾 Storing embeddings in ChromaDB...")
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"   ✅ Stored {len(embeddings)} embeddings")
    
    def test_retrieval(self, query: str, n_results: int = 3):
        print(f"\n🔍 Testing retrieval with query:")
        print(f"   '{query}'")
        print(f"\n   Retrieving top {n_results} results...\n")
        
        query_embedding = self.model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        print("="*80)
        print("📊 RETRIEVAL RESULTS")
        print("="*80)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = 1 - distance
            
            print(f"\n🎯 Result #{i}")
            print(f"   Source: {metadata['source']}.pdf")
            print(f"   Chunk: {metadata['chunk_index']}")
            print(f"   Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
            print(f"   Length: {metadata['char_count']} chars")
            print(f"\n   📝 Content:")
            preview = doc[:300] + "..." if len(doc) > 300 else doc
            print(f"   {preview}")
            print("-"*80)
    
    def print_summary(self):
        count = self.collection.count()
        
        print("\n" + "="*80)
        print("📊 EMBEDDING PIPELINE SUMMARY")
        print("="*80)
        print(f"Total embeddings stored: {count}")
        print(f"Embedding model: {EMBEDDING_MODEL}")
        print(f"Vector database: ChromaDB")
        print(f"Database location: {CHROMA_DB_DIR}")
        print(f"Collection name: {COLLECTION_NAME}")
        print(f"\n✅ Ready for RAG queries!")
        print("="*80 + "\n")


def main():
    pipeline = EmbeddingPipeline()
    
    pipeline.load_embedding_model()
    
    chunks = pipeline.load_chunks()
    
    pipeline.initialize_chromadb()
    
    pipeline.generate_and_store_embeddings(chunks)
    
    pipeline.print_summary()
    
    print("\n🧪 Running test queries...\n")
    
    test_queries = [
        "What is Fermi energy?",
        "Explain Hall effect",
        "What are the properties of carbon nanotubes?"
    ]
    
    for query in test_queries:
        pipeline.test_retrieval(query, n_results=2)
        print("\n")


if __name__ == "__main__":
    main()