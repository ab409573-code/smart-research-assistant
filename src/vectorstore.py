import logging
import os
from pathlib import Path
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def store_documents_in_pinecone(chunks):
    """
    Store document chunks in Pinecone vector store.
    
    Args:
        chunks: List of LangChain Document objects
    
    Returns:
        VectorStore instance
    
    Raises:
        ValueError: If environment variables are missing or API calls fail
    """
    try:
        # Validate environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "smart-research")
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        logger.info(f"🔄 Initializing Pinecone with index: {index_name}")
        
        # Initialize embeddings
        logger.info("🔄 Loading Google Generative AI embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Pinecone
        pc = PineconeClient(api_key=pinecone_api_key)
        
        # Store documents
        logger.info(f"🔄 Storing {len(chunks)} chunks in Pinecone...")
        vector_store = PineconeVectorStore.from_documents(
            chunks,
            embeddings,
            index_name=index_name
        )
        
        logger.info(f"✓ Successfully stored {len(chunks)} chunks in Pinecone")
        return vector_store
        
    except KeyError as e:
        logger.error(f"❌ Missing environment variable: {e}")
        raise ValueError(f"Missing environment variable: {e}")
    except Exception as e:
        logger.error(f"❌ Error storing documents in Pinecone: {type(e).__name__}: {str(e)}", exc_info=True)
        raise

def query_pinecone(query_text: str, top_k: int = 5):
    """
    Query documents from Pinecone.
    
    Args:
        query_text: User's question
        top_k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    try:
        logger.info(f"🔍 Querying Pinecone: '{query_text}'")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        index_name = os.getenv("PINECONE_INDEX_NAME", "smart-research")
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        
        results = vector_store.similarity_search(query_text, k=top_k)
        logger.info(f"✓ Found {len(results)} relevant documents")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error querying Pinecone: {type(e).__name__}: {str(e)}", exc_info=True)
        raise