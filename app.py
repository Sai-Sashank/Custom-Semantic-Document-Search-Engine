from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import math
from collections import defaultdict
from typing import List, Dict

app = FastAPI(title="Custom TF-IDF Semantic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR = "documents"

# Globals
vocabulary: List[str] = []
vocab_index: Dict[str, int] = {}
doc_vectors: Dict[str, List[float]] = {}
doc_titles: Dict[str, str] = {}
doc_snippets: Dict[str, str] = {}
idf_values: Dict[str, float] = {}

def tokenize(text: str) -> List[str]:
    """Clean and tokenize text"""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = text.split()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'this', 'that', 'these', 'those'
    }
    return [t for t in tokens if t not in stop_words and len(t) > 2]

def build_index():
    global vocabulary, vocab_index, doc_vectors, doc_titles, doc_snippets, idf_values

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        print(f"Created {DOCS_DIR} folder.")
        return

    doc_tokens = {}
    raw_documents = {}

    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith('.txt'):
            continue
            
        path = os.path.join(DOCS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            raw_documents[filename] = content
            
            title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
            doc_titles[filename] = title

            content_tokens = tokenize(content)
            title_tokens = tokenize(title)
            
            # Stronger title boost
            combined = content_tokens + (title_tokens * 4)   # Increased from 3 to 4
            doc_tokens[filename] = combined

    if not doc_tokens:
        print("No documents found.")
        return

    # Vocabulary
    all_words = set()
    for tokens in doc_tokens.values():
        all_words.update(tokens)
    
    vocabulary = sorted(all_words)
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    N = len(doc_tokens)

    # Document Frequency
    df = defaultdict(int)
    for tokens in doc_tokens.values():
        for word in set(tokens):
            df[word] += 1

    # Improved IDF
    idf_values = {word: math.log((N + 1) / (df[word] + 0.5)) + 1 for word in vocabulary}

    # TF-IDF Vectors
    doc_vectors = {}
    for filename, tokens in doc_tokens.items():
        tf = defaultdict(int)
        for word in tokens:
            tf[word] += 1

        vector = [0.0] * len(vocabulary)
        doc_len = max(len(tokens), 1)
        
        for word, count in tf.items():
            if word in vocab_index:
                idx = vocab_index[word]
                tf_val = count / doc_len
                vector[idx] = tf_val * idf_values.get(word, 0)
        
        doc_vectors[filename] = vector

    # Snippets
    doc_snippets = {}
    for filename, content in raw_documents.items():
        lines = content.strip().split('\n')
        snippet = lines[0] if lines else content[:200]
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        doc_snippets[filename] = snippet

    print(f"✅ Indexed {len(doc_vectors)} documents | Vocabulary: {len(vocabulary)} terms")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Manual Cosine Similarity"""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def query_to_vector(query: str) -> List[float]:
    tokens = tokenize(query)
    if not tokens or not vocabulary:
        return [0.0] * len(vocabulary)

    tf = defaultdict(int)
    for word in tokens:
        tf[word] += 1

    vector = [0.0] * len(vocabulary)
    query_len = len(tokens)

    for word, count in tf.items():
        if word in vocab_index:
            idx = vocab_index[word]
            tf_val = count / query_len
            vector[idx] = tf_val * idf_values.get(word, 1.0)
    
    return vector


@app.on_event("startup")
async def startup_event():
    build_index()


@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    if not doc_vectors:
        return {"results": []}

    query_vec = query_to_vector(q)
    if sum(abs(x) for x in query_vec) < 0.0001:
        return {"results": []}

    scored_docs = []
    for filename, doc_vec in doc_vectors.items():
        score = cosine_similarity(query_vec, doc_vec)
        scored_docs.append((filename, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    results = []
    for filename, score in scored_docs[:3]:
        results.append({
            "document": filename,
            "title": doc_titles.get(filename, filename),
            "score": round(float(score), 4),
            "snippet": doc_snippets.get(filename, "")
        })

    return {"results": results}


@app.get("/index")
def reindex():
    build_index()
    return {"status": "re-indexed", "documents": len(doc_vectors)}


@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)