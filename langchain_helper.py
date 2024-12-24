from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import Field
import torch
import os
import hashlib

from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH')
VECTORS_PATH = os.getenv('VECTORS_PATH')

if not all([MODEL_PATH, DOCUMENTS_PATH, VECTORS_PATH]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

class LocalLlamaLLM(LLM):
    model_path: str
    device: str = Field(default="cuda")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="auto",
            torch_dtype=torch.float16
        )

    @property
    def _llm_type(self) -> str:
        return "local_llama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_documents_hash():
    """Generate a hash of all PDFs in the documents folder"""
    hash_md5 = hashlib.md5()
    
    for filename in sorted(os.listdir(DOCUMENTS_PATH)):
        if filename.endswith('.pdf'):
            filepath = os.path.join(DOCUMENTS_PATH, filename)
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def process_pdf_folder(folder_path):
    """Process all PDFs from a specified folder."""
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    return split_documents(documents)

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def get_vectorstore(texts=None, force_refresh=False):
    """Get or create vector store"""
    os.makedirs(VECTORS_PATH, exist_ok=True)
    
    # Generate hash of current documents
    current_hash = get_documents_hash()
    hash_file_path = os.path.join(VECTORS_PATH, "docs_hash")
    vector_store_path = os.path.join(VECTORS_PATH, "faiss_index")
    
    regenerate = force_refresh or not os.path.exists(vector_store_path)
    if not regenerate and os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as f:
            stored_hash = f.read().strip()
            if stored_hash != current_hash:
                regenerate = True
    
    if regenerate:
        print("Generating new vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
        
        if texts is None:
            texts = process_pdf_folder(DOCUMENTS_PATH)
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        vectorstore.save_local(vector_store_path)
        with open(hash_file_path, 'w') as f:
            f.write(current_hash)
    else:
        print("Loading existing vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
        vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
    return vectorstore

def setup_qa_chain(force_refresh=False):
    """Set up the question-answering chain with local embeddings and LLM."""
    try:
        vectorstore = get_vectorstore(force_refresh=force_refresh)
        
        print("Loading LLaMA model...")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LocalLlamaLLM(
            model_path=MODEL_PATH,
            callback_manager=callback_manager
        )
        
        print("Setting up QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        raise Exception(f"Error setting up QA chain: {str(e)}")