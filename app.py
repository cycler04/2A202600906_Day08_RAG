"""
RAG Chatbot Application - Streamlit UI

Ứng dụng chatbot dùng để trả lời câu hỏi về pháp luật ma tuý và các bài báo liên quan.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.task10_generation import generate_with_citation

# Page config
st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("⚖️ Chatbot Pháp Luật Ma Tuý")
st.write("Hỏi đáp về luật phòng chống ma tuý và các tin tức liên quan")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    top_k = st.slider(
        "Số lượng tài liệu cần lấy ra:",
        min_value=1,
        max_value=10,
        value=5,
        help="Càng nhiều tài liệu, câu trả lời càng chi tiết nhưng cũng chậm hơn"
    )
    
    use_reranking = st.checkbox(
        "Sử dụng reranking",
        value=True,
        help="Reranking giúp sắp xếp lại kết quả để tìm những tài liệu liên quan nhất"
    )
    
    st.markdown("---")
    st.subheader("📊 Thông tin hệ thống")
    st.info("""
    **Kiến trúc RAG:**
    - Semantic Search (Dense Retrieval)
    - Lexical Search (BM25)
    - RRF (Reciprocal Rank Fusion)
    - Reranking
    - Fallback: PageIndex Vectorless
    
    **LLM:** OpenAI GPT-4o-mini
    
    **Dữ liệu:** Pháp luật Việt Nam + Tin tức
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Tài liệu tham khảo"):
                for i, source in enumerate(message["sources"][:3], 1):
                    st.markdown(f"""
                    **[{i}] {source.get('metadata', {}).get('source', 'Unknown')}**
                    
                    {source['content'][:300]}...
                    """)

# Chat input
user_input = st.chat_input("Hỏi câu hỏi của bạn...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            try:
                result = generate_with_citation(user_input, top_k=top_k)
                
                answer = result["answer"]
                sources = result["sources"]
                retrieval_source = result["retrieval_source"]
                
                st.write(answer)
                
                # Display retrieval source
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nguồn Retrieval", retrieval_source.upper())
                with col2:
                    st.metric("Số Tài Liệu", len(sources))
                
                # Display sources
                with st.expander("📚 Tài liệu tham khảo"):
                    for i, source in enumerate(sources, 1):
                        source_name = source.get("metadata", {}).get("source", "Unknown")
                        st.markdown(f"""
                        **[{i}] {source_name}**
                        
                        {source['content'][:300]}...
                        """)
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": answer,
                    "sources": sources,
                    "retrieval_source": retrieval_source
                })
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý: {str(e)}")
                st.info("💡 Hãy chắc chắn rằng bạn đã:")
                st.write("1. Chạy `python -m src.task4_chunking_indexing` để tạo index")
                st.write("2. Đặt OPENAI_API_KEY trong file .env (hoặc sẽ dùng mock response)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>RAG Pipeline for Vietnamese Drug Law | Powered by OpenAI GPT-4o-mini</p>
</div>
""", unsafe_allow_html=True)
