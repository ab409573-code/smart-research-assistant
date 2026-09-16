import streamlit as st
import requests

API_BASE_URL = "https://smart-research-assistant-4p6i.onrender.com/api"

st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Smart Research Assistant")
st.caption("Production-ready RAG Assistant powered by FastAPI, ChromaDB, and Gemini")

# الشريط الجانبي لرفع الملفات
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Indexing and extracting chunks..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"File indexed! Total Chunks: {data.get('chunks_count')}")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# إدارة سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📍 Citations & Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **Source:** `{s.get('source')}` | **Page:** `{s.get('page')}`")

# إدخال السؤال
if prompt := st.chat_input("Ask a question about your documents..."):
    # عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # إرسال السؤال إلى API
    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving context..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"question": prompt, "top_k": 3}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📍 Citations & Sources"):
                            for s in sources:
                                st.markdown(f"- **Source:** `{s.get('source')}` | **Page:** `{s.get('page')}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")