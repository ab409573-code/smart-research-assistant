import os
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def ask_research_assistant(question: str, top_k: int = 3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "smart-research")
    
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # استخدام Gemini للإجابة
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa_chain.invoke({"query": question})
    
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
            
    return {
        "answer": result["result"],
        "sources": sources
    }