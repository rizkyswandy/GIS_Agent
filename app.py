import streamlit as st
import os
from langchain_helper import setup_qa_chain
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH')
VECTORS_PATH = os.getenv('VECTORS_PATH')

def check_environment():
    """Check if all required paths and files exist"""
    if not os.path.exists(VECTORS_PATH):
        st.error("Vectors directory not found. Please run initialize_vectors.py first.")
        return False
    
    vector_store_path = os.path.join(VECTORS_PATH, "faiss_index")
    if not os.path.exists(vector_store_path):
        st.error("Vector store not found. Please run initialize_vectors.py first.")
        return False
    
    return True

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        try:
            with st.spinner("Loading QA system..."):
                st.session_state.qa_chain = setup_qa_chain()
                st.success("✅ System loaded successfully")
        except Exception as e:
            st.error(f"Error loading system: {str(e)}")
            return False
    return True

def main():
    st.set_page_config(page_title="PDF Knowledge Base Chat", layout="wide")
    st.title("📚 PDF Knowledge Base Chat")

    if not check_environment():
        st.stop()

    with st.sidebar:
        st.header("Available Documents")
        pdfs = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
        st.success(f"Found {len(pdfs)} PDFs in documents folder")
        if pdfs:
            with st.expander("View Documents"):
                for pdf in pdfs:
                    st.text(f"• {pdf}")
        
        if st.button("🔄 Start New Chat"):
            st.session_state.messages = []
            st.rerun()

    if not initialize_chat_history():
        st.stop()

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                result = st.session_state.qa_chain({"query": prompt})
                response = result['result']

                if "Helpful Answer:" in response:
                    response = response.split("Helpful Answer:", 1)[1].strip()

                st.write(response)

                
                with st.expander("View Sources"):
                    for doc in result['source_documents']:
                        st.markdown(f"**From page {doc.metadata['page']}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()