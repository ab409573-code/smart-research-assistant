import os
from typing import List
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_with_pages(pdf_path: str) -> List[Document]:
    """
    استخراج النصوص من كل صفحة في الـ PDF مع حفظ اسم الملف ورقم الصفحة في Metadata
    """
    documents = []
    file_name = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_number
                    }
                )
                documents.append(doc)
    return documents


def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Document]:
    """
    تقطيع النصوص إلى أجزاء أصغر مع الحفاظ على ترابط المعنى والـ Metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    print("Ingestion module is ready!")