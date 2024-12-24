from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import hashlib

load_dotenv()

DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH')
VECTORS_PATH = os.getenv('VECTORS_PATH')

def process_pdfs():
    """Process all PDFs and generate vector store"""
    print("Starting PDF processing...")
    
    # Create vectors directory if it doesn't exist
    os.makedirs(VECTORS_PATH, exist_ok=True)
    
    # Load all PDFs
    documents = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith('.pdf'):
            print(f"Processing {file}...")
            pdf_path = os.path.join(DOCUMENTS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    # Split documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    # Generate embeddings and store
    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    # Create and save vector store
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save vector store
    vector_store_path = os.path.join(VECTORS_PATH, "faiss_index")
    print(f"Saving vector store to {vector_store_path}...")
    vectorstore.save_local(vector_store_path)
    
    # Generate and save document hash
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