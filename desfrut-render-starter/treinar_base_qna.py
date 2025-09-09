import os, csv, uuid
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/data/chroma")
COLLECTION = "qna"
MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")  # alinhado com seu app.py
API_KEY = os.environ["OPENAI_API_KEY"]

def train(csv_path: str):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=API_KEY, model_name=MODEL)
    col = client.get_or_create_collection(COLLECTION, embedding_function=ef)
    docs, metas, ids = [], [], []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("pergunta") or "").strip()
            a = (row.get("resposta") or "").strip()
            t = (row.get("tags") or "").strip()
            if not q or not a:
                continue
            docs.append(q + "\n\nRESPOSTA:\n" + a)
            metas.append({"tags": t})
            ids.append(str(uuid.uuid4()))
    if docs:
        col.add(documents=docs, metadatas=metas, ids=ids)
        print(f"✅ Inseridos {len(docs)} itens em '{COLLECTION}' em {CHROMA_DIR}")
    else:
        print("Nada para inserir. Verifique o CSV.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python treinar_base_qna.py base_qna.csv")
        raise SystemExit(1)
    train(sys.argv[1])
