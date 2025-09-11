# app.py — Desfrut IA (Estável): Agente + Q&A + RAG + /admin/train (sem "Fontes")

from flask import Flask, request, jsonify, render_template_string
import os, json, re, csv, difflib
from datetime import datetime

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ========= ENV =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEN_MODEL      = os.getenv("GEN_MODEL", "gpt-4o-mini")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "/data/chroma")
COL_APOSTILA   = os.getenv("COL_APOSTILA", "desfrut_apostila")
COL_PRODUTOS   = os.getenv("COL_PRODUTOS", "desfrut_produtos")
TOP_K          = int(os.getenv("TOP_K", "5"))
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
STATE_DB       = os.getenv("STATE_DB", "/data/state.json")
PRODUTOS_CSV   = os.getenv("PRODUTOS_CSV", "produtos.csv")
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", "")  # defina em Environment

# ========= APP =========
app = Flask(__name__)
oai  = OpenAI(api_key=OPENAI_API_KEY)

# ========= Saudação/humanização (Manaus) =========
try:
    import pytz
    TZ = pytz.timezone("America/Manaus")
except Exception:
    TZ = None

def _saudacao_periodo():
    try:
        agora = datetime.now(TZ) if TZ else datetime.now()
        h = agora.hour
        if   5 <= h < 12:  return "Bom dia! "
        elif 12 <= h < 18: return "Boa tarde! "
        else:              return "Boa noite! "
    except Exception:
        return "Oi! "

def humanize(texto: str, nome: str | None = None) -> str:
    texto = (texto or "").strip()
    if not texto:
        return "Posso te ajudar com mais alguma coisa?"
    prefix = _saudacao_periodo()
    if nome:
        prefix = prefix.replace("!", f", {nome.split(' ')[0]}! ")
    primeiros = texto[:20].lower()
    if any(p in primeiros for p in ["oi", "olá", "ola", "boa "]):
        prefix = ""
    return (prefix + texto).strip()

# ========= Embeddings/Chroma =========
def embed_one(text: str):
    emb = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

def get_collection(name: str):
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name)

def retrieve(col_name: str, question: str, top_k: int = TOP_K):
    col = get_collection(col_name)
    qvec = embed_one(question)
    res = col.query(query_embeddings=[qvec], n_results=top_k, include=["documents", "metadatas"])
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas

# ========= RAG (apostila + produtos) =========
def build_context(question: str):
    parts = []
    try:
        d1, m1 = retrieve(COL_APOSTILA, question)
        if d1:
            parts.append("=== APOSTILA ===")
            for d, _ in zip(d1, m1):
                parts.append(d)
    except Exception as e:
        parts.append(f"(Apostila indisponível: {e})")
    try:
        d2, m2 = retrieve(COL_PRODUTOS, question)
        if d2:
            parts.append("\n=== PRODUTOS ===")
            for d, _ in zip(d2, m2):
                parts.append(d)
    except Exception as e:
        parts.append(f"(Produtos indisponíveis: {e})")
    return "\n\n".join(parts)

def answer_rag(question: str) -> str:
    ctx = build_context(question)
    if not ctx.strip():
        user_content = (f"Pergunta: {question}\n\n"
                        "Contexto (vazio). Diga que não encontrou na base e ofereça uma orientação geral breve.")
    else:
        user_content = (f"Pergunta: {question}\n\n"
                        f"Contexto (use com prioridade):\n{ctx}")
    resp = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content":
             "Você é a assistente da Desfrut (sexshop em Manaus). "
             "Responda acolhedor, objetivo e educativo. Priorize o contexto fornecido. "
             "Se não houver no contexto, diga isso e dê orientação geral breve. "
             "Evite conteúdo explícito. Para compras, direcione ao site/Tray."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ========= Q&A base (coleção 'qna') =========
def answer_qna(query: str):
    try:
        col = get_collection("qna")
        qvec = embed_one(query)
        r = col.query(query_embeddings=[qvec], n_results=1, include=["documents", "distances"])
        if not r or not r.get("documents") or not r["documents"][0]:
            return None
        doc  = r["documents"][0][0]
        dist = r["distances"][0][0]  # menor = mais parecido
        if dist <= 0.20:
            if "RESPOSTA:" in doc:
                return doc.split("RESPOSTA:\n", 1)[-1].strip()
            return doc.strip()
        return None
    except Exception:
        return None

# ========= Memória + Produtos CSV (Agente) =========
def _load_state():
    try:
        with open(STATE_DB, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(d):
    try:
        with open(STATE_DB, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _carregar_produtos():
    itens = []
    try:
        with open(PRODUTOS_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                itens.append({k.lower(): v for k, v in r.items()})
    except Exception:
        pass
    return itens

PROD_CACHE = None
def buscar_produto(termo: str, n=3):
    global PROD_CACHE
    if PROD_CACHE is None:
        PROD_CACHE = _carregar_produtos()
    if not PROD_CACHE:
        return []
    termo = (termo or "").strip()
    # SKU
    for p in PROD_CACHE:
        if termo.lower() in str(p.get('sku','')).lower():
            return [p]
    # Fuzzy nome
    nomes = [p.get('nome') or p.get('título') or p.get('titulo') or '' for p in PROD_CACHE]
    match = difflib.get_close_matches(termo, nomes, n=n, cutoff=0.5)
    res = [p for p in PROD_CACHE if (p.get('nome') or p.get('título') or p.get('titulo') or '') in match]
    return res[:n]

CEP_RE = re.compile(r"\b\d{5}-?\d{3}\b")
def tool_cotar_frete(cep: str):
    cep = re.sub(r"\D", "", cep)
    if cep.startswith("690"):
        return "Em Manaus, oferecemos frete imediato grátis em horário comercial. Informe o bairro para estimativa do tempo de entrega."
    return ("Para o seu CEP, coto frete pelos Correios (PAC/Sedex). Me diga a cidade/UF e se deseja entrega econômica (PAC) "
            "ou rápida (Sedex). Se preferir, posso estimar com peso padrão (0,5 kg).")

def tool_ver_produto(termo: str):
    itens = buscar_produto(termo, n=3)
    if not itens:
        return "Não encontrei esse item agora. Pode me enviar o nome exato ou o SKU?"
    linhas = []
    for p in itens:
        nome    = p.get('nome') or p.get('título') or p.get('titulo') or 'Produto'
        sku     = p.get('sku') or '—'
        preco   = p.get('preco') or p.get('preço') or p.get('valor') or 'sob consulta'
        estoque = p.get('estoque') or p.get('qtd') or p.get('quantidade') or ''
        if estoque:
            linhas.append(f"• {nome} (SKU {sku}) – R$ {preco} — estoque: {estoque}")
        else:
            linhas.append(f"• {nome} (SKU {sku}) – R$ {preco}")
    linhas.append("Se quiser, já separo no seu nome. Me diga o SKU ou a opção que gostou.")
    return "\n".join(linhas)

def tool_criar_pedido(state: dict):
    pedido_id = f"DFT-{str(abs(hash(json.dumps(state))) % 100000).zfill(5)}"
    return (f"Pedido rascunho criado: {pedido_id}. Agora me confirme o método de pagamento (Pix ou cartão em até 6x) "
            f"e o endereço/retirada para eu finalizar.")

def agente_responder(user_text: str, customer_id: str | None, customer_name: str | None):
    txt = (user_text or '').strip()
    if not txt:
        return None
    db = _load_state()
    st = db.get(customer_id or 'anon', {"carrinho": []})

    m = CEP_RE.search(txt)
    if m:
        cep = m.group(0)
        st['cep'] = cep
        db[customer_id or 'anon'] = st
        _save_state(db)
        return tool_cotar_frete(cep)

    gatilhos = ["tem ", "estoque", "disponível", "preço", "valor", "sku", "tamanho", "cor"]
    if any(g in txt.lower() for g in gatilhos):
        termo = re.sub(r"\b(tem|de|o|a|um|uma|preço|valor|do|da|no|na|sku|tamanho|cor|disponível|estoque)\b", "", txt, flags=re.I)
        termo = termo.strip() or txt
        return tool_ver_produto(termo)

    if re.search(r"\b(finalizar|fechar|comprar|fechar pedido|checkout)\b", txt, flags=re.I):
        return tool_criar_pedido(st)

    return None

# ========= Admin: treino de Q&A via HTTP (sem shell) =========
def train_qna(csv_path: str, reset: bool = False) -> int:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection("qna")
        except Exception:
            pass
    # Sem embedding_function aqui
    col = client.get_or_create_collection("qna")

    docs, metas, ids, vecs = [], [], [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("pergunta") or "").strip()
            a = (row.get("resposta") or "").strip()
            t = (row.get("tags") or "").strip()
            if not q or not a:
                continue
            doc = q + "\n\nRESPOSTA:\n" + a
            docs.append(doc)
            metas.append({"tags": t})
            ids.append(os.urandom(8).hex())
            # embed do lado de cá: usa sua função embed_one (OpenAI embeddings)
            vecs.append(embed_one(q))

    if not docs:
        return 0

    col.add(documents=docs, metadatas=metas, ids=ids, embeddings=vecs)
    return len(docs)

@app.get('/admin/train')
def admin_train():
    token = request.args.get('token', '')
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error='forbidden'), 403
    path = request.args.get('path', 'base_qna.csv')
    reset = request.args.get('reset', '0').lower() in ('1','true','yes','on')
    try:
        n = train_qna(path, reset=reset)
        return jsonify(ok=True, inserted=n, reset=reset, path=path)
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=path), 500

# ========= Páginas =========
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
    #q { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
    #send { padding: 10px 16px; border-radius: 8px; border: 0; background: #111; color: #fff; cursor: pointer; margin-top: 8px; }
    #send:disabled { opacity: .5; }
    #answer { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Desfrut IA</h1>
  <p>Faça sua pergunta com base na apostila, Q&A e catálogo de produtos.</p>
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
    document.getElementById('out').innerHTML = `<div id="answer"><b>IA:</b> ${ans}</div>`;
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

@app.get("/healthz")
def healthz():
    return jsonify(ok=True)

# ========= /ask =========
@app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}
    user_q    = (data.get("question") or "").strip()
    cust_id   = data.get("customer_id")
    cust_name = data.get("customer_name")
    if not user_q:
        return jsonify({"error": "Pergunta vazia."}), 400

    # 1) Q&A
    qna = answer_qna(user_q)
    if qna:
        return jsonify({"answer": humanize(qna, cust_name)})

    # 2) Agente (ferramentas)
    agent_ans = agente_responder(user_q, cust_id, cust_name)
    if agent_ans:
        return jsonify({"answer": humanize(agent_ans, cust_name)})

    # 3) RAG fallback
    try:
        ans = answer_rag(user_q)
        return jsonify({"answer": humanize(ans, cust_name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========= MAIN =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
