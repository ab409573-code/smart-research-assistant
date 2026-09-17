import os
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def store_documents_in_pinecone(chunks):
    # استخدام نماذج جوجل الخفيفة بدلاً من الموديلات المحلية الثقيلة
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME", "smart-research") # تأكد إن اسم الاندكس يطابق اللي عندك
    
    Pinecone.from_documents(chunks, embeddings, index_name=index_name)