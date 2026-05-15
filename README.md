# Chat-With-Your-Data (RAG) Assistant

A modern, model-agnostic Retrieval-Augmented Generation (RAG) application built with LangChain, LiteLLM, ChromaDB, and Streamlit. This app allows users to upload PDF documents and ask questions based strictly on the document's content, while dynamically supporting over 100+ LLMs (Anthropic, Gemini, Groq, Ollama, Cohere, etc.).

## 🌟 Features
- **Model Agnostic**: Powered by `litellm`, switch between top-tier models (Claude 3.5 Sonnet, Gemini Pro) or local open-source models seamlessly via the UI.
- **Conversational Memory**: Built-in history-aware retrieval ensures the AI remembers the context of your conversation for natural follow-up questions.
- **Secure by Design**: API keys are passed securely at runtime via the sidebar and never hardcoded or saved to files.
- **Local Vector Database**: Uses ChromaDB to efficiently embed and search through your private documents completely locally.

## 🚀 Quick Start

### 1. Set up the virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Orchestration**: LangChain (`langchain-classic`, `langchain-core`)
- **LLM Routing**: LiteLLM
- **Embeddings & Vector Store**: ChromaDB
- **Document Parsing**: PyPDF
