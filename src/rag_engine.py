import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.documents import Document

try:
    from src.vectorstore import get_vectorstore
except ModuleNotFoundError:
    from vectorstore import get_vectorstore

load_dotenv()

# تهيئة إعدادات Gemini
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def format_docs(docs: List[Document]) -> str:
    """تنسيق المستندات المسترجعة مع أرقام الصفحات والمصدر"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def ask_research_assistant(question: str, top_k: int = 3) -> Dict[str, Any]:
    # 1. استرجاع المقاطع الأكثر ملائمة من قاعدة المتجهات ChromaDB
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    matched_docs = retriever.invoke(question)
    
    # 2. إعداد السياق والاستشهادات
    context = format_docs(matched_docs)
    sources = [
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        for doc in matched_docs
    ]
    
    # 3. صياغة التوجيه للسياق فقط
    prompt = f"""You are an expert research assistant. Answer the question accurately using ONLY the provided context.
If the answer cannot be found in the context, say clearly: "I cannot find this information in the provided document."

Context:
{context}

Question: {question}

Helpful Answer:"""

    # 4. توليد الإجابة بالنموذج المدعوم حالياً بحسابك
    model = genai.GenerativeModel("gemini-3.6-flash")
    response = model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "sources": sources
    }

if __name__ == "__main__":
    test_question = "What career tracks or skills are mentioned?"
    print(f"Testing RAG Engine with question: '{test_question}'...\n")
    try:
        result = ask_research_assistant(test_question)
        print("Answer:\n", result["answer"])
        print("\nCitations/Sources:\n", result["sources"])
    except Exception as e:
        print("Error:", e)