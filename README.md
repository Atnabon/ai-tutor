# 📘 AI Tutor 

AI Tutor is a Streamlit web application that allows students and educators to upload their own textbooks or notes (PDFs), ask questions, and receive AI-generated answers and summaries grounded in their materials. The app uses LangChain for document processing, retrieval, and LLM-powered generation.

---

## Features

- **Document Ingestion:** Upload multiple PDF textbooks or notes. Files are automatically chunked for efficient retrieval.
- **Indexing & Storage:** Text chunks are embedded using Sentence Transformers and stored in a Chroma vector database.
- **Retrieval-Augmented Generation (RAG):** Ask questions about your uploaded materials and receive answers with references.
- **Summarization:** Select any uploaded file and get a concise summary of its content.
- **User-Friendly Interface:** Interact via a simple Streamlit web app.

---

## Getting Started

### Prerequisites

- Python 3.9+
- [pip](https://pip.pypa.io/en/stable/)
- Groq API key (for LLM access)

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/atnabon/ai-tutor.git
   cd textbook-tutor
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   # Or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key:**
so    - Create a `.env` file in the project root:

       ```env
       GROQ_API_KEY=your_groq_api_key_here
       # (Optional) Override default lightweight model
       # See <https://console.groq.com/docs/models> for available models
       GROQ_MODEL=llama-3.1-8b-instant
       ```

---

## Usage

1. **Start the Streamlit app:**

   ```sh
   streamlit run src/app_streamlit.py
   ```

2. **Upload your textbooks/notes:**
   - Use the sidebar to upload PDF files.
   - **Important:** Name your files using the format `Grade <number> <Subject>`, e.g., `Grade 9 Biology.pdf`.

3. **Ask questions or request summaries:**
   - Enter your question in the main page and click "Get Answer".
   - Or, select a file and click "Summarize" for a summary.

---

## File Structure

```txt
textbook-tutor/
│
├── src/
│   ├── app_streamlit.py      # Streamlit web app
│   ├── qa_pipeline.py        # Document processing, retrieval, and chains
│   └── ...                   # Other source files
├── requirements.txt
├── .env                      # Your Groq API key
└── README.md
```

---

## Technologies Used

- [Streamlit](https://streamlit.io/) - Web app framework
- [LangChain](https://python.langchain.com/) - Document loaders, chains, prompts
- [Chroma](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Groq](https://groq.com/) - LLM API

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgements

- LangChain documentation and community
- Streamlit community
- Groq API

---

## Troubleshooting

- **Groq API Key Error:** Make sure your `.env` file is present and contains a valid `GROQ_API_KEY`.
- **Model Decommissioned Error:** If you see an error like `model_decommissioned`, set a supported model in `.env` via `GROQ_MODEL` (e.g., `llama-3.1-8b-instant` or a larger model like `llama-3.1-70b-versatile`).
- **PDF Loading Issues:** Ensure your PDFs are not password-protected and are named correctly.
- **Module Import Errors:** Double-check your Python environment and installed packages.
- **File Watch Performance:** Install `watchdog` for faster Streamlit reloads:


   ```sh
   pip install watchdog
   ```

- **Tokenizer Parallelism Warning:** We set `TOKENIZERS_PARALLELISM=false` automatically to avoid fork warnings. You can override this in your shell if desired.

---

## 🚀 Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (ensure `.env` is NOT committed – it is already ignored via `.gitignore`).
2. Go to <https://streamlit.io/cloud> and choose Deploy → select the repo.
3. Set the entry point to: `src/app_streamlit.py`.
4. Add your secrets (App → Settings → Secrets) in TOML format:

   ```toml
   GROQ_API_KEY = "your_real_groq_key"
   # Optional: override model
   GROQ_MODEL = "llama-3.1-8b-instant"
   ```

5. Click Deploy. On first run, upload PDFs to start asking questions.

### How Secrets Are Loaded

Code checks for (in order):

1. Environment variables (`GROQ_API_KEY`, `GROQ_MODEL`)
2. `st.secrets["GROQ_API_KEY"]` / `st.secrets["GROQ_MODEL"]`
3. Falls back to default model `llama-3.1-8b-instant` if none given.

### Changing the Model Later

Just update the `GROQ_MODEL` value in Secrets and reboot the app. Examples:

- Faster & cheaper: `llama-3.1-8b-instant`
- Better reasoning: `llama-3.1-70b-versatile`

---

## 🗄️ Vector Store Persistence

Currently, the Chroma vector store is created in-memory / ephemeral each session. After a redeploy or container restart, users must re-upload PDFs.

Options to persist:

1. Local ephemeral directory with `persist_directory="chroma_db"` (works only for current container lifetime).
2. Switch to a managed vector DB (Pinecone, Qdrant Cloud, Weaviate) for durable storage.
3. Add simple caching keyed by file hash to avoid re-embedding identical uploads.

If you want help implementing any of these, open an issue or request an enhancement.

---
