import os
import chromadb
from utils.embedding_utils import get_embedding, load_json
from dotenv import load_dotenv
from typing import List, Dict, Any
import time

# Load environment variables
load_dotenv()

# Paths
DATA_PATH = "data/cleaned_products.json"
CHROMA_DIR = "vector_store/chroma_db"

def create_product_text(product: Dict[str, Any]) -> str:
    """
    Creates a formatted text representation of a product for embedding.
    """
    cover_types = product.get('coverType', [])
    cover_type_str = ', '.join(cover_types) if isinstance(cover_types, list) else str(cover_types)
    
    return f"""Product Name: {product.get('productName', '')}
        Brand: {product.get('brandName', '')}
        Model: {product.get('phoneModel', '')}
        Type: {cover_type_str}
        Description: {product.get('productDescription', '')}
        Price: {product.get('productPrice', '')}
        Category: {product.get('productCategory', '')}"""

def initialize_chroma_client():
    """
    Initializes and returns a Chroma client with proper settings.
    """
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    # Initialize with persistent client (newer ChromaDB syntax)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    return client

def embed_and_store_products(batch_size: int = 10):
    """
    Embeds products and stores them in ChromaDB with batch processing.
    """
    try:
        # Load products
        print(f"📂 Loading products from {DATA_PATH}...")
        products = load_json(DATA_PATH)
        print(f"✅ Loaded {len(products)} products")
        
        # Initialize Chroma client
        print(f"\n🔧 Initializing ChromaDB at {CHROMA_DIR}...")
        client = initialize_chroma_client()
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="products",
            metadata={"description": "E-commerce product embeddings"}
        )
        
        print(f"✅ Collection ready: {collection.name}")
        print(f"\n🚀 Starting embedding process...")
        print(f"{'='*60}")
        
        success_count = 0
        error_count = 0
        
        for idx, product in enumerate(products):
            try:
                # Create product text
                text = create_product_text(product)
                
                # Generate embedding
                embedding = get_embedding(text)
                
                if embedding is None:
                    print(f"⚠️  Skipped product {idx+1}/{len(products)}: Empty embedding")
                    error_count += 1
                    continue
                
                # Prepare metadata
                metadata = {
                    "productName": str(product.get("productName", "")),
                    "productUrl": str(product.get("productUrl", "")),
                    "productImage": str(product.get("productImage", "")),
                    "productPrice": str(product.get("productPrice", "")),
                    "brandName": str(product.get("brandName", "")),
                    "phoneModel": str(product.get("phoneModel", "")),
                    "productCategory": str(product.get("productCategory", ""))
                }
                
                # Add to collection
                collection.add(
                    ids=[f"product_{idx}"],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text]
                )
                
                success_count += 1
                product_name = product.get('productName', 'Unknown')[:50]
                print(f"✅ [{success_count}/{len(products)}] {product_name}")
                
                # Rate limiting - avoid API throttling
                if (idx + 1) % batch_size == 0:
                    print(f"⏸️  Processed {idx + 1} products, pausing briefly...")
                    time.sleep(1)
                
            except Exception as e:
                error_count += 1
                print(f"❌ Error processing product {idx+1}: {str(e)}")
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 Embedding Complete!")
        print(f"✅ Successfully embedded: {success_count}")
        print(f"❌ Errors: {error_count}")
        print(f"📊 Total products: {len(products)}")
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
    print("🤖 E-commerce Product Embedding Pipeline")
    print("="*60)
    
    # Run embedding process
    success, errors = embed_and_store_products(batch_size=10)
    
    if success > 0:
        print("\n✨ Ready to use ChromaDB for product search!")
    else:
        print("\n⚠️  No products were successfully embedded.")