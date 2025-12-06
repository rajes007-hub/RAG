import pickle
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sklearn.preprocessing import normalize

chunks = pickle.load(open("data/chunks.pkl", "rb"))
embeddings = np.load("data/embeddings.npy")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path="rag/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048, temperature=0.1, verbose=False)

app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    user_query = request.json.get("query", "")
    q_vec = normalize(embed_model.encode([user_query]), axis=1)
    _, idx = index.search(q_vec, 3)
    context = "\n\n".join(chunks[i] for i in idx[0])
    prompt = f"Answer the question based on the context below.\n\n{context}\n\nQuestion: {user_query}\nAnswer:"
    result = llm(prompt, max_tokens=200, stop=["\n"])
    return jsonify({"answer": result["choices"][0]["text"].strip()})

if __name__ == "__main__":
    app.run(port=5000)
