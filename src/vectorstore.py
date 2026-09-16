import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import EMBEDDING_MODEL_NAME

# تحميل المتغيرات من ملف .env
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore():
    # الاتصال بقاعدة البيانات السحابية
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embedding_function(),
        pinecone_api_key=PINECONE_API_KEY
    )

def store_documents_in_pinecone(chunks):
    # رفع الملفات الجديدة إلى السحابة
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore