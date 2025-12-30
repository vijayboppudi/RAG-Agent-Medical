# 🩺 Medical RAG Assistant

A beautiful and intelligent Retrieval-Augmented Generation (RAG) chatbot for medical information, built with FastAPI, OpenAI, Qdrant, and LangGraph.

## 🚀 Features

- **Beautiful Modern UI**: Sleek chat interface with gradient backgrounds and smooth animations
- **RAG Pipeline**: Retrieval-augmented generation using vector search
- **Medical Focus**: Specialized for health and medical information queries
- **Real-time Chat**: Interactive chatbot with typing indicators
- **Source Citations**: Shows sources and confidence scores for answers
- **Responsive Design**: Works on desktop and mobile

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **AI/ML**: OpenAI GPT-4, OpenAI Embeddings, LangGraph
- **Vector Database**: Qdrant
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Jupyter Notebooks

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key
- Virtual environment (recommended)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/medical-rag-assistant.git
   cd medical-rag-assistant
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn openai qdrant-client langgraph python-dotenv
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 📊 Data Setup

1. **Prepare your documents**: Place medical documents in the `docs/` folder
2. **Run data processing notebooks**:
   - `ingest_normalize.ipynb` - Process and normalize documents
   - `chunk_docs.ipynb` - Create document chunks
   - `embed_load_chunk.ipynb` - Generate embeddings and load to Qdrant

## 🚀 Running the Application

1. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Open your browser** and go to: `http://127.0.0.1:8000`

3. **Start chatting** with your medical RAG assistant!

## 📁 Project Structure

```
medical-rag-assistant/
├── main.py              # FastAPI application
├── index.html           # Beautiful chat UI
├── docs/               # Medical documents
├── normalized/         # Processed document chunks
├── qdrant_local/       # Vector database (excluded from git)
├── *.ipynb            # Data processing notebooks
├── requirements.txt    # Dependencies (to be created)
├── .env               # Environment variables (excluded from git)
└── README.md          # This file
```

## 🤖 How It Works

1. **Document Processing**: Medical documents are processed and chunked
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings stored in Qdrant vector database
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **RAG Generation**: Retrieved relevant chunks are used as context for GPT-4 responses
6. **Beautiful UI**: Modern chat interface displays responses with source citations

## 🔒 Safety Features

- Medical disclaimer and safety warnings
- Advice to consult healthcare professionals
- Emergency situation detection and guidance
- Source citation for transparency

## 📝 License

MIT License - feel free to use for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**⚠️ Medical Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical concerns.