
import os, re, sys, time, traceback
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
PDF_FILENAME = os.getenv("PDF_FILENAME", os.path.join("data", "apostila.pdf"))

def log(msg): print(msg, flush=True)

try:
    log("Iniciando treinamento (apostila)...")
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    if not os.path.exists(PDF_FILENAME):
        raise FileNotFoundError(f"Não achei {PDF_FILENAME}. Coloque seu PDF em data/apostila.pdf")

    reader = PdfReader(PDF_FILENAME)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            pages.append((i, t))
        else:
            log(f"⚠️ Página {i} sem texto (pode ser PDF escaneado).")

    def chunk_text(text, size=1000, overlap=150):
        out, start, L = [], 0, len(text)
        while start < L:
            end = min(L, start + size)
            out.append(text[start:end])
            start = max(end - overlap, end)
        return out

    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(name="desfrut_apostila", metadata={"hnsw:space":"cosine"})

    total = 0
    for page_num, text in pages:
        parts = chunk_text(text)
        for j, part in enumerate(parts):
            emb = client.embeddings.create(model=EMBED_MODEL, input=part)
            vec = emb.data[0].embedding
            collection.add(
                ids=[f"p{page_num:03d}-c{j:03d}"],
                documents=[part],
                embeddings=[vec],
                metadatas=[{"file": os.path.basename(PDF_FILENAME), "page": page_num}]
            )
            total += 1
            if total % 20 == 0: log(f"🔧 Processados {total} pedaços...")

    log(f"✅ Treinamento concluído! Itens salvos: {total}")
    log(f"📦 Base: {os.path.abspath(CHROMA_DIR)}")

except Exception as e:
    print("\nERRO CAPTURADO:")
    traceback.print_exc()
    print("\nVerifique env vars (OPENAI_API_KEY, CHROMA_DIR) e o arquivo em data/apostila.pdf")
