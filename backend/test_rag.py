import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "student_mentor_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print(" Loading model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("  Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

print(f" Ready! Database has {collection.count()} embeddings\n")

def ask_question(query: str, n_results: int = 3):
    print(f"\n Question: {query}")
    print("="*80)
    
    query_embedding = model.encode([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = (1 - distance) * 100
        
        print(f"\n Result #{i} - {similarity:.1f}% match")
        print(f" Source: {metadata['source']}.pdf (Chunk {metadata['chunk_index']})")
        print(f" Content:\n{doc[:400]}...")
        print("-"*80)

print("\n" + "="*80)
print(" INTELLIGENT STUDENT MENTOR - RAG SYSTEM TEST")
print("="*80)

while True:
    print("\n Ask a question (or 'quit' to exit):")
    query = input(">>> ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("\n Goodbye!")
        break
    
    if not query:
        continue
    
    ask_question(query, n_results=3)