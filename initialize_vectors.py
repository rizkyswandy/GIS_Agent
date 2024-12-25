from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import hashlib
import logging

load_dotenv()

DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH')
VECTORS_PATH = os.getenv('VECTORS_PATH')

def verify_vectorstore(vectorstore, embeddings):
    test_query = "test"
    test_embedding = embeddings.embed_query(test_query)
    try:
        vectorstore.similarity_search_by_vector(test_embedding)
        return True
    except AssertionError:
        logging.error("Vector dimension mismatch")
        return False
    except Exception as e:
        logging.error(f"Error verifying vector store: {e}")
        return False

def process_pdfs():
    """Process all PDFs and generate vector store"""
    print("Starting PDF processing...")
    
    os.makedirs(VECTORS_PATH, exist_ok=True)
    
    documents = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith('.pdf'):
            print(f"Processing {file}...")
            pdf_path = os.path.join(DOCUMENTS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            for page in pages:
                page.metadata.update({
                    "source": file,
                    "page": page.metadata.get("page", 0) + 1
                })
            documents.extend(pages)
    
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-t5-xl",
        model_kwargs={'device': 'cuda'}
    )
    
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    if not verify_vectorstore(vectorstore, embeddings):
        raise RuntimeError("Vector store verification failed")
    
    vector_store_path = os.path.join(VECTORS_PATH, "faiss_index")
    print(f"Saving vector store to {vector_store_path}...")
    vectorstore.save_local(vector_store_path)
    
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(DOCUMENTS_PATH)):
        if filename.endswith('.pdf'):
            filepath = os.path.join(DOCUMENTS_PATH, filename)
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_md5.update(chunk)
    
    hash_file_path = os.path.join(VECTORS_PATH, "docs_hash")
    with open(hash_file_path, 'w') as f:
        f.write(hash_md5.hexdigest())
    
    print("Vector store initialization complete!")

if __name__ == "__main__":
    process_pdfs()