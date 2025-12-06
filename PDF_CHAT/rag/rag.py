import os
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sklearn.preprocessing import normalize

def load_pdf(filepath):
    reader = PdfReader(filepath)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, normalize(np.array(embeddings), axis=1)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, chunks, model, index, k=6, min_score=0.3, max_tokens=1000):
    query_vec = normalize(model.encode([query]), axis=1)
    scores, indices = index.search(np.array(query_vec), k)
    selected_chunks = []
    token_count = 0

    for score, idx in zip(scores[0], indices[0]):
        if score < min_score:
            continue
        chunk = chunks[idx]
        chunk_tokens = len(chunk.split())
        if token_count + chunk_tokens > max_tokens:
            break
        selected_chunks.append(chunk)
        token_count += chunk_tokens

    return selected_chunks

def generate_answer(query, chunks, model, index, mistral_model_path):
    relevant_chunks = retrieve_relevant_chunks(query, chunks, model, index)
    context = "\n\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context below.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    llm = Llama(model_path=mistral_model_path, n_ctx=2048, verbose=False, temperature=0.1)
    output = llm(prompt, max_tokens=200, stop=["\n"])
    return output["choices"][0]["text"].strip()
def main():
    doc_path = "data/company_acko.pdf"
    mistral_model_path = "rag/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    text = load_pdf(doc_path)
    chunks = chunk_text(text)
    model, embeddings = generate_embeddings(chunks)

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save("data/embeddings.npy", embeddings)

    index = build_faiss_index(embeddings)

    while True:
        query = input("\nEnter your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = generate_answer(query, chunks, model, index, mistral_model_path)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()