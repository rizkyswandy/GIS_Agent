# GIS Based Agent

An interactive chat application that allows users to have conversations about their PDF documents using a local LLaMA model. The application uses LangChain for document processing and embeddings, and Streamlit for the user interface.

## Features

- Chat-based interface for asking questions about your PDFs
- Uses local LLaMA model for question answering
- Maintains conversation history for contextual responses
- Document source references with page numbers
- Easy-to-use web interface
- Support for multiple PDF documents
- Vector store caching for faster responses

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (for running LLaMA model, I'm using 4 RTX 2080TI)
- LLaMA model downloaded locally

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gis-agent.git
cd gis-agent

2. Create and activate a virtual environment:
```bash
python -m venv .langchain
source .langchain/bin/activate  # On Windows: .langchain\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your paths:
```env
MODEL_PATH="/path/to/your/llama/model"
DOCUMENTS_PATH="/path/to/your/documents"
VECTORS_PATH="/path/to/your/vectors"
```

## Project Structure

```
pdf-knowledge-chat/
├── app.py                 # Streamlit web application
├── langchain_helper.py    # LangChain and LLM implementation
├── documents/            # Directory for PDF files
├── vectors/             # Directory for vector store
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

1. Place your PDF documents in the `documents` folder.

2. Initialize the vector store (first time only):
```bash
python initialize_vectors.py
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

5. Start chatting with your documents!

## Features in Detail

### Vector Store Initialization
- Processes PDF documents into chunks
- Creates embeddings using sentence-transformers
- Stores vectors for efficient retrieval

### Conversation Management
- Maintains chat history
- Provides contextual responses
- Supports follow-up questions

### Document References
- Shows source documents for answers
- Includes page numbers for easy reference
- Expandable source view

## Dependencies

- langchain
- langchain-community
- streamlit
- transformers
- torch
- python-dotenv
- sentence-transformers
- faiss-cpu (or faiss-gpu)
- pypdf

## Configuration

You can modify the following parameters in the code:

- Chunk size for document splitting (default: 1000)
- Chunk overlap (default: 200)
- Number of relevant chunks for answering (default: 3)
- Model parameters in the LlamaModel class

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://streamlit.io/)
- [LLaMA](https://github.com/facebookresearch/llama)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)