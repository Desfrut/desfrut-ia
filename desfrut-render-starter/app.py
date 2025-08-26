
import os
from flask import Flask, request, jsonify, render_template_string
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ===== Env vars =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
COL_APOSTILA = os.getenv("COL_APOSTILA", "desfrut_apostila")
COL_PRODUTOS = os.getenv("COL_PRODUTOS", "desfrut_produtos")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

app = Flask(__name__)

def get_collection(name: str):
    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name)

def retrieve(col_name: str, question: str, top_k: int = TOP_K):
    col = get_collection(col_name)
    oai = OpenAI(api_key=OPENAI_API_KEY)
    emb = oai.embeddings.create(model=EMBED_MODEL, input=question)
    qvec = emb.data[0].embedding
    res = col.query(query_embeddings=[qvec], n_results=top_k, include=["documents", "metadatas"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas

def build_context(question: str):
    # busca nas duas coleções (apostila + produtos)
    context_parts = []
    fontes = []

    try:
        docs_a, metas_a = retrieve(COL_APOSTILA, question)
        if docs_a:
            context_parts.append("=== APOSTILA ===")
            for d, m in zip(docs_a, metas_a):
                context_parts.append(d)
                fontes.append(f"{m.get('file','apostila.pdf')} pág. {m.get('page','?')}")
    except Exception as e:
        context_parts.append(f"(Apostila indisponível: {e})")

    try:
        docs_p, metas_p = retrieve(COL_PRODUTOS, question)
        if docs_p:
            context_parts.append("\n=== PRODUTOS ===")
            for d, m in zip(docs_p, metas_p):
                context_parts.append(d)
                sku = m.get("sku") or "SKU?"
                nome = m.get("nome") or "Produto"
                fontes.append(f"{nome} ({sku})")
    except Exception as e:
        context_parts.append(f"(Produtos indisponíveis: {e})")

    context = "\n\n".join(context_parts)
    fontes = list(dict.fromkeys(fontes))
    return context, fontes

def answer(question: str):
    context, fontes = build_context(question)
    oai = OpenAI(api_key=OPENAI_API_KEY)

    if not context.strip():
        user_content = (f"Pergunta: {question}\n\n"
                        "Contexto (vazio). Diga que não encontrou na base e ofereça uma orientação geral breve.")
    else:
        user_content = (f"Pergunta: {question}\n\n"
                        f"Contexto (use com prioridade, cite quando útil):\n{context}")

    resp = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content":
             "Você é a assistente da Desfrut (sexshop em Manaus). "
             "Responda acolhedor, objetivo e educativo. Priorize as evidências do contexto. "
             "Se não houver no contexto, diga isso e dê orientação geral breve. "
             "Evite conteúdo explícito. Para compras, direcione ao site/Tray."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    return text, fontes

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Desfrut IA</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 16px; }
    h1 { margin-bottom: 8px; }
    .box { background: #f8f8f8; padding: 16px; border-radius: 12px; }
    .msg { margin: 12px 0; }
    .src { font-size: 12px; color: #666; margin-top: 8px; }
    input, button, textarea { font-size: 16px; }
    #q { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
    #send { padding: 10px 16px; border-radius: 8px; border: 0; background: #111; color: #fff; cursor: pointer; margin-top: 8px; }
    #send:disabled { opacity: .5; }
    #answer { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Desfrut IA</h1>
  <p>Faça sua pergunta com base na apostila e no catálogo de produtos.</p>
  <div class="box">
    <textarea id="q" rows="3" placeholder="Digite sua pergunta..."></textarea>
    <br/>
    <button id="send">Perguntar</button>
  </div>

  <div id="out" class="msg"></div>

<script>
async function ask() {
  const btn = document.getElementById('send');
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  btn.disabled = true; btn.textContent = "Perguntando...";
  document.getElementById('out').innerHTML = "";

  try {
    const res = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q})});
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    const ans = data.answer || "(sem resposta)";
    const fontes = (data.fontes || []).join("; ");

    document.getElementById('out').innerHTML =
      `<div id="answer"><b>IA:</b> ${ans}</div>` +
      (fontes ? `<div class="src"><b>Fontes:</b> ${fontes}</div>` : "");
  } catch(e) {
    document.getElementById('out').textContent = "Erro: " + e;
  } finally {
    btn.disabled = false; btn.textContent = "Perguntar";
  }
}
document.getElementById('send').onclick = ask;
document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
});
</script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error":"Pergunta vazia."}), 400
    try:
        ans, fontes = answer(q)
        return jsonify({"answer": ans, "fontes": fontes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
