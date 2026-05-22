# Custom Semantic Document Search Engine

A lightweight backend application that performs semantic search over a corpus of text documents using **pure TF-IDF + Cosine Similarity**, implemented from scratch without any external NLP libraries.

---

## Features

- Manual TF-IDF vectorization (fully custom implementation)
- Manual Cosine Similarity computation
- Filename-based title boosting (4x weight)
- Clean REST API using FastAPI
- Minimal responsive UI
- Re-indexing support (`/index` endpoint)
- Returns top 3 most relevant documents with scores and snippets

## Tech Stack

- **Python 3.10+**
- **FastAPI** (web framework)
- **Uvicorn** (ASGI server)
- Only standard libraries: `re`, `math`, `collections`, `os`

---

## Project Structure
├── app.py # Main backend with TF-IDF + API\
├── templates\
├──    └── index.html # Minimal frontend UI\
├── documents # Folder containing 50+ .txt files\
├── README.md\
└── requirements.txt

---

## Setup Instructions

1. **Clone / Create project folder**
   ```bash
   mkdir custom-semantic-search && cd custom-semantic-search
   
   git clone https://github.com/Sai-Sashank/Custom-Semantic-Document-Search-Engine.git
3. **Create documents folder and add files**:
   ```bash
   mkdir documents
**Copy all 50 .txt files into the "documents/" folder (or use the dataset included in the page)**

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn
4. **Run the application**:
   ```bash
   uvicorn app:app --reload

5. **Open browser**:
   ```bash
   http://localhost:8000

---

1. **Search Documents**:
   GET /search?q=your search query 
   
   **Example**:
   ```bash
   curl -G --data-urlencode "q=tv future in the hands of viewers" http://localhost:8000/search

2. **If you want to see a clean structured response use jq**:
   ```bash
   sudo apt install jq

   curl -s -G "http://localhost:8000/search" --data-urlencode "q=artificial intelligence in finance " | jq

3. **You can also search using the search box in the UI. Just type what you want in the text box.**\
   Example: artificial intelligence in finance
   ```bash
   http://localhost:8000
   
## Design Decisions & Implementation Details
1. Text Processing:

   Custom tokenize() function using regex.\
   Basic stop-word removal.\
   Lowercasing + punctuation cleaning

3. Vectorization (TF-IDF)

   Term Frequency (TF): Normalized by document length\
   Inverse Document Frequency (IDF): Smoothed logarithmic formula\
   Title Boosting: Filename is tokenized and given 4x weight to improve relevance

3. Similarity

   Cosine Similarity implemented manually using dot product and vector norms

4. Why Scores May Appear Moderate\
   With a large vocabulary (~4000 terms) and news-style documents, absolute cosine scores tend to be lower. The relative ranking is what matters, and the system correctly identifies the most relevant document as rank #1.

## Future Improvements (Not Implemented due to constraints)

BM25 ranking algorithm\
Query expansion\
Stemming / Lemmatization\
Better snippet extraction around query terms
