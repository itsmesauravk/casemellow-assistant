import os
import chromadb
from utils.embedding_utils import get_embedding, load_json
from dotenv import load_dotenv
from typing import List, Dict, Any
import time

# Load environment variables
load_dotenv()

# Paths
DATA_PATH = "data/faqs.json"
CHROMA_DIR = "vector_store/chroma_db"

def create_faq_text(faq: Dict[str, Any]) -> str:
    """
    Creates a formatted text representation of a FAQ for embedding.
    """
    question = faq.get("question", "")
    answer = faq.get("answer", "")
    return f"Q: {question}\nA: {answer}"

def initialize_chroma_client():
    """
    Initializes and returns a Chroma client with proper settings.
    """
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    return client

def embed_and_store_faqs(batch_size: int = 10):
    """
    Embeds FAQs and stores them in ChromaDB with batch processing.
    """
    try:
        # Load FAQs
        print(f"📂 Loading FAQs from {DATA_PATH}...")
        faqs = load_json(DATA_PATH)
        print(f"✅ Loaded {len(faqs)} FAQs")
        
        # Initialize Chroma client
        print(f"\n🔧 Initializing ChromaDB at {CHROMA_DIR}...")
        client = initialize_chroma_client()
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="faqs",
            metadata={"description": "FAQ embeddings for e-commerce chatbot"}
        )
        
        print(f"✅ Collection ready: {collection.name}")
        print(f"\n🚀 Starting embedding process...")
        print(f"{'='*60}")
        
        success_count = 0
        error_count = 0
        
        for idx, faq in enumerate(faqs):
            try:
                # Create text for embedding
                text = create_faq_text(faq)
                
                # Generate embedding
                embedding = get_embedding(text)
                
                if embedding is None:
                    print(f"⚠️  Skipped FAQ {idx+1}/{len(faqs)}: Empty embedding")
                    error_count += 1
                    continue
                
                # Prepare metadata
                metadata = {
                    "question": str(faq.get("question", "")),
                    "answer": str(faq.get("answer", ""))
                }
                
                # Add to collection
                collection.add(
                    ids=[f"faq_{idx}"],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text]
                )
                
                success_count += 1
                print(f"✅ [{success_count}/{len(faqs)}] FAQ: {faq.get('question', '')[:50]}")
                
                # Rate limiting - avoid API throttling
                if (idx + 1) % batch_size == 0:
                    print(f"⏸️  Processed {idx + 1} FAQs, pausing briefly...")
                    time.sleep(1)
                
            except Exception as e:
                error_count += 1
                print(f"❌ Error processing FAQ {idx+1}: {str(e)}")
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 Embedding Complete!")
        print(f"✅ Successfully embedded: {success_count}")
        print(f"❌ Errors: {error_count}")
        print(f"📊 Total FAQs: {len(faqs)}")
        print(f"💾 Database location: {CHROMA_DIR}")
        print(f"{'='*60}")
        
        # Verify collection
        collection_count = collection.count()
        print(f"\n🔍 Verification: Collection contains {collection_count} embeddings")
        
        return success_count, error_count
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_PATH}")
        print("Please ensure the data file exists.")
        return 0, 0
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0

if __name__ == "__main__":
    print("🤖 FAQ Embedding Pipeline")
    print("="*60)
    
    # Run embedding process
    success, errors = embed_and_store_faqs(batch_size=10)
    
    if success > 0:
        print("\n✨ Ready to use ChromaDB for FAQ search!")
    else:
        print("\n⚠️  No FAQs were successfully embedded.")
