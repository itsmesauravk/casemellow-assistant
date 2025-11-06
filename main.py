import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.embedding_utils import get_embedding
import chromadb
from google import genai
from dotenv import load_dotenv
import json
from fastapi.middleware.cors import CORSMiddleware
from embed_products import embed_and_store_products
from embed_faqs import embed_and_store_faqs



# Load environment variables
load_dotenv()

# -------------------- CONFIG --------------------
CHROMA_DIR = "vector_store/chroma_db"
TOP_K_PRODUCTS = 3
TOP_K_FAQS = 2
MAX_RESPONSE_TOKENS = 1000

# -------------------- FASTAPI --------------------
app = FastAPI(
    title="Casemellow Chatbot API",
    version="2.0",
    description="RAG-powered e-commerce chatbot with Gemini"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods: GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)

# -------------------- MODELS --------------------
class QueryRequest(BaseModel):
    query: str
    top_k_products: Optional[int] = TOP_K_PRODUCTS
    top_k_faqs: Optional[int] = TOP_K_FAQS
    user_id: Optional[str] = None  # For conversation history tracking

class ProductResult(BaseModel):
    productName: str
    productUrl: str
    productImage: str
    productPrice: str
    brandName: Optional[str] = None
    phoneModel: Optional[str] = None

class FAQResult(BaseModel):
    question: str
    answer: str

class QueryResponse(BaseModel):
    query: str
    responseText: str
    products: List[ProductResult]
    faqs: List[FAQResult]
    hasResults: bool
    conversationContext: Optional[str] = None

# -------------------- GEMINI SETUP --------------------
try:
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print("✅ Gemini API initialized")
except Exception as e:
    print(f"❌ Error initializing Gemini: {str(e)}")
    gemini_client = None

# -------------------- CHROMA SETUP --------------------
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    print("Adding products and FAQs collections...")
    embed_and_store_products()
    embed_and_store_faqs()
    print("✅ Embedding scripts completed")
    products_collection = client.get_collection("products")
    faqs_collection = client.get_collection("faqs")
    
    print(f"✅ ChromaDB loaded from {CHROMA_DIR}")
    print(f"📦 Products: {products_collection.count()} embeddings")
    print(f"❓ FAQs: {faqs_collection.count()} embeddings")
    
except Exception as e:
    print(f"❌ Error loading ChromaDB: {str(e)}")
    products_collection = None
    faqs_collection = None

# -------------------- HELPER FUNCTIONS --------------------
def retrieve_products(query_text: str, top_k: int = TOP_K_PRODUCTS):
    """Retrieve relevant products from ChromaDB"""
    if products_collection is None:
        return []
    
    try:
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            return []
        
        results = products_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 10),  # Max 10
            include=['metadatas', 'distances']
        )
        
        products = []
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                # Filter: only include products with required fields
                if metadata.get("productName") and metadata.get("productPrice"):
                    products.append(ProductResult(
                        productName=metadata.get("productName", ""),
                        productUrl=metadata.get("productUrl", ""),
                        productImage=metadata.get("productImage", ""),
                        productPrice=metadata.get("productPrice", ""),
                        brandName=metadata.get("brandName"),
                        phoneModel=metadata.get("phoneModel")
                    ))
        
        return products
    
    except Exception as e:
        print(f"❌ Error retrieving products: {str(e)}")
        return []

def retrieve_faqs(query_text: str, top_k: int = TOP_K_FAQS):
    """Retrieve relevant FAQs from ChromaDB"""
    if faqs_collection is None:
        return []
    
    try:
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            return []
        
        results = faqs_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 5),  # Max 5
            include=['metadatas', 'distances']
        )
        
        faqs = []
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                if metadata.get("question") and metadata.get("answer"):
                    faqs.append(FAQResult(
                        question=metadata.get("question", ""),
                        answer=metadata.get("answer", "")
                    ))
        
        return faqs
    
    except Exception as e:
        print(f"❌ Error retrieving FAQs: {str(e)}")
        return []

def generate_response_with_gemini(query: str, products: List[ProductResult], faqs: List[FAQResult]):
    """Generate conversational response using Gemini with retrieved context"""
    if gemini_client is None:
        return "I'm having trouble connecting to the AI service. Please try again later."
    
    # Build context from retrieved data
    products_context = ""
    if products:
        products_context = "Available Products:\n"
        for i, prod in enumerate(products[:5], 1):  # Limit to top 5 for context
            products_context += f"{i}. {prod.productName} - {prod.productPrice}\n"
            if prod.brandName:
                products_context += f"   Brand: {prod.brandName}\n"
            if prod.phoneModel:
                products_context += f"   Model: {prod.phoneModel}\n"
    
    faqs_context = ""
    if faqs:
        faqs_context = "\nFrequently Asked Questions:\n"
        for i, faq in enumerate(faqs[:3], 1):  # Limit to top 3
            faqs_context += f"Q{i}: {faq.question}\nA{i}: {faq.answer}\n\n"
    
    # Construct prompt
    prompt = f"""You are a helpful e-commerce chatbot assistant for Casemellow, a phone case store.

User Query: "{query}"

{products_context}

{faqs_context}

Instructions:
- Provide a friendly, conversational response to the user's query
- If products are available, mention them naturally (e.g., "I found some great options for you...")
- If FAQs are relevant, incorporate that information naturally
- If no products/FAQs are found, politely suggest alternative searches or general help
- Keep response concise (2-4 sentences)
- Be enthusiastic and helpful
- Don't mention technical terms like "embeddings" or "database"
- Don't make up product details not in the context

Response:"""

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config={
                'temperature': 0.7,
                'max_output_tokens': MAX_RESPONSE_TOKENS,
                'top_p': 0.9,
            }
        )
        
        return response.text.strip()
    
    except Exception as e:
        print(f"❌ Error generating response: {str(e)}")
        # Fallback response
        if products:
            return f"I found {len(products)} product(s) that match your query. Check them out below!"
        elif faqs:
            return f"I found some helpful information that might answer your question."
        else:
            return "I couldn't find exactly what you're looking for. Try rephrasing your question or browse our categories!"

# -------------------- API ENDPOINTS --------------------
@app.post("/query", response_model=QueryResponse)
def query_chatbot(request: QueryRequest):
    """
    Main chatbot endpoint with RAG pipeline:
    1. Retrieve relevant products and FAQs
    2. Generate conversational response with Gemini
    3. Return structured response for frontend
    """
    query_text = request.query.strip()
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if len(query_text) > 500:
        raise HTTPException(status_code=400, detail="Query is too long. Max 500 characters.")
    
    if products_collection is None or faqs_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Please contact support."
        )
    
    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable. Please try again."
        )
    
    try:
        # Step 1: Retrieve relevant products
        products = retrieve_products(
            query_text, 
            top_k=min(request.top_k_products, 10)
        )
        
        # Step 2: Retrieve relevant FAQs
        faqs = retrieve_faqs(
            query_text,
            top_k=min(request.top_k_faqs, 5)
        )
        
        # Step 3: Generate conversational response with Gemini
        response_text = generate_response_with_gemini(query_text, products, faqs)
        
        # Step 4: Build response
        has_results = len(products) > 0 or len(faqs) > 0
        
        return QueryResponse(
            query=query_text,
            responseText=response_text,
            products=products,
            faqs=faqs,
            hasResults=has_results,
            conversationContext=f"Found {len(products)} products and {len(faqs)} FAQs"
        )
    
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again."
        )

# -------------------- HEALTH & INFO --------------------
@app.get("/health")
def healthcheck():
    """Check API health and service status"""
    return {
        "status": "ok",
        "services": {
            "chromadb": products_collection is not None and faqs_collection is not None,
            "gemini": gemini_client is not None
        },
        "collections": {
            "products": products_collection.count() if products_collection else 0,
            "faqs": faqs_collection.count() if faqs_collection else 0
        }
    }

@app.get("/")
def root():
    """API information"""
    return {
        "message": "Casemellow RAG Chatbot API",
        "version": "2.0",
        "features": [
            "Vector similarity search",
            "RAG with Gemini AI",
            "Product recommendations",
            "FAQ assistance"
        ],
        "endpoints": {
            "POST /query": "Ask the chatbot",
            "GET /health": "Check API status",
            "GET /docs": "Interactive API docs"
        }
    }

# -------------------- STARTUP --------------------
@app.on_event("startup")
def startup_event():
    """Validate services on startup"""
    print("\n" + "="*60)
    print("🚀 Casemellow Chatbot API Starting...")
    print("="*60)
    
    if products_collection is None or faqs_collection is None:
        print("⚠️  WARNING: ChromaDB not loaded!")
        print("Run embedding scripts before starting API.")
    
    if gemini_client is None:
        print("⚠️  WARNING: Gemini API not initialized!")
        print("Check your GOOGLE_API_KEY in .env file.")
    
    if products_collection and faqs_collection and gemini_client:
        print("✅ All services ready!")
    
    print("="*60 + "\n")