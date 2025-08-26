
import os, csv, re
from openai import OpenAI
import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
CSV_PATH = os.getenv("CSV_PATH", os.path.join("data", "produtos.csv"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
COLLECTION_NAME = os.getenv("COL_PRODUTOS", "desfrut_produtos")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def clean(t): 
    t = t or ""
    return re.sub(r"\s+", " ", str(t)).strip()

def detect_dialect(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            return csv.Sniffer().sniff(sample, delimiters=";,|\\t,")
        except Exception:
            class D: delimiter = ','; quotechar = '"'
            return D()

def row_get(row, *keys):
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return ""

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Não encontrei {CSV_PATH}. Coloque seu CSV em data/produtos.csv")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    col = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})

    dialect = detect_dialect(CSV_PATH)
    ids, texts, metas = [], [], []

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for i, row in enumerate(reader, start=1):
            nome = clean(row_get(row, "nome","Nome","name","Name","Título","titulo"))
            sku  = clean(row_get(row, "sku","SKU","codigo","Código","code","Code"))
            preco = clean(row_get(row, "preco","preço","price","Price"))
            desc = clean(row_get(row, "descricao","Descrição","description","Description"))
            cat  = clean(row_get(row, "categoria","Categoria","categoria1","Categoria1"))
            est  = clean(row_get(row, "estoque","Estoque","stock","Stock"))

            if not nome and not desc:
                continue

            doc = (
                f"PRODUTO\\n"
                f"Nome: {nome}\\n"
                f"SKU: {sku}\\n"
                f"Preço: {preco}\\n"
                f"Categoria: {cat}\\n"
                f"Estoque: {est}\\n"
                f"Descrição: {desc}"
            )
            ids.append(f"prod-{i:06d}")
            texts.append(doc)
            metas.append({"sku": sku, "nome": nome})

    BATCH = 64
    total = 0
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i+BATCH]
        batch_ids   = ids[i:i+BATCH]
        batch_meta  = metas[i:i+BATCH]
        emb = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
        vecs = [d.embedding for d in emb.data]
        col.add(ids=batch_ids, documents=batch_texts, embeddings=vecs, metadatas=batch_meta)
        total += len(batch_texts)
        print(f"✅ Indexados {total}/{len(texts)}")

    print(f"🎉 Produtos indexados! Total: {total}. Coleção: {COLLECTION_NAME}. Base: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
