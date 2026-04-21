# Alternate Reality Engine (ARE)

Alternate Reality Engine is a Streamlit app for causal RAG and counterfactual simulation:

- Upload domain text knowledge
- Extract causal relationships from retrieved context
- Build and visualize a causal graph
- Simulate "what-if this event never happened?" scenarios

Live app: [alternate-reality-5x7wwldjd3ssmxqykdpleq.streamlit.app](https://alternate-reality-5x7wwldjd3ssmxqykdpleq.streamlit.app/)

## Why deployment showed "Database error"

Your deployed app failed during the ChromaDB initialization path. Streamlit Cloud environments can be stricter about native/runtime dependencies and package compatibility, so code that works locally may fail there.

This project now uses an **in-memory vector store** instead of ChromaDB, so no external database is required for this app flow.

## Project Structure

- `main.py` - Streamlit UI and pipeline orchestration
- `modules/ingestion.py` - chunking, embeddings, in-memory retrieval
- `modules/causal_extraction.py` - LLM-based causal pair extraction
- `modules/graph_builder.py` - graph construction and visualization
- `modules/counterfactual.py` - counterfactual reasoning and explanations

## Local Run Instructions

### 1) Clone and open project

```bash
git clone <your-repo-url>
cd "Alternate-Reality"
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit app

```bash
streamlit run main.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How to Use

1. Enter your Groq API key in the sidebar
2. Upload a `.txt` file in Step 1
3. Click **Store in Knowledge Base**
4. Enter a causal query in Step 2 and click **Analyze Causal Chain**
5. Explore graph in Step 3 and simulate counterfactuals in Step 4

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. In Streamlit Cloud, create a new app from this repo
3. Set:
   - Main file path: `main.py`
   - Python version: 3.10+ (recommended 3.10 or 3.11)
4. Deploy

No database setup is required.

## Troubleshooting

- If model download is slow on first run, wait for initial embedding model load
- If API calls fail, verify your Groq API key
- If memory usage is high, use smaller input files (the store is in memory)

## Tech Stack

- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- NetworkX + Matplotlib
- Groq-compatible OpenAI SDK client
