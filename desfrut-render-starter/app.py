# app.py — Desfrut IA (resiliente + agente melhorado)
# Agente + Q&A + RAG + /admin/train + /admin/qna_count. Sem “Fontes”. Nunca 500 no /ask.

from flask import Flask, request, jsonify, render_template_string
import os, json, re, csv, difflib
from datetime import datetime

# ===== Config/env =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEN_MODEL      = os.getenv("GEN_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "/tmp/chroma")      # mude p/ /data/chroma se tiver Disk
STATE_DB       = os.getenv("STATE_DB", "/tmp/state.json")    # mude p/ /data/state.json se tiver Disk
PRODUTOS_CSV   = os.getenv("PRODUTOS_CSV", "base_produtos.csv")
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", "")

app = Flask(__name__)

# ===== OpenAI (usa se houver chave) =====
oai = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        oai = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        oai = None

# ===== Timezone p/ saudação =====
try:
    import pytz
    TZ = pytz.timezone("America/Manaus")
except Exception:
    TZ = None

def _saudacao():
    try:
        h = (datetime.now(TZ) if TZ else datetime.now()).hour
        if 5 <= h < 12:  return "Bom dia! "
        if 12 <= h < 18: return "Boa tarde! "
        return "Boa noite! "
    except Exception:
        return "Oi! "

def humanize(texto: str, nome: str | None = None) -> str:
    texto = (texto or "").strip()
    if not texto:
        return "Posso te ajudar com mais alguma coisa?"
    pref = _saudacao()
    if nome:
        pref = pref.replace("!", f", {nome.split(' ')[0]}! ")
    if texto[:20].lower().startswith(("oi", "olá", "ola", "boa ")):
        pref = ""
    return (pref + texto).strip()

# ========= Utils: OpenAI embeddings com fallback silencioso =========
def _embed(text: str):
    if not oai: return None
    try:
        e = oai.embeddings.create(model=EMBED_MODEL, input=text)
        return e.data[0].embedding
    except Exception:
        return None

# ========= Produtos CSV =========
def _carregar_produtos():
    itens = []
    try:
        with open(PRODUTOS_CSV, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
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
    # SKU primeiro
    for p in PROD_CACHE:
        if termo.lower() in str(p.get("sku","")).lower():
            return [p]
    nomes = [p.get('nome') or p.get('título') or p.get('titulo') or '' for p in PROD_CACHE]
    match = difflib.get_close_matches(termo, nomes, n=n, cutoff=0.5)
    res = [p for p in PROD_CACHE if (p.get('nome') or p.get('título') or p.get('titulo') or '') in match]
    return res[:n]

# ========= Memória simples =========
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

# ========= Ferramentas do agente =========
CEP_RE = re.compile(r"\b\d{5}-?\d{3}\b")
# NOVO: intenção de entrega/frete mesmo sem CEP
ENTREGA_RE = re.compile(r"\b(entrega|entregam|entregas|frete|envia|enviam|delivery|motoboy|retirada|retira|buscar)\b", re.I)

def tool_cotar_frete(cep: str):
    cep = re.sub(r"\D", "", cep)
    if cep.startswith("690"):
        return "Em Manaus, oferecemos frete imediato grátis em horário comercial. Informe o bairro para estimativa do tempo de entrega."
    return ("Para o seu CEP, coto frete pelos Correios (PAC/Sedex). Me diga a cidade/UF e se prefere PAC (econômico) ou Sedex (rápido).")

def tool_politica_entrega_sem_cep():
    return ("Atendemos Manaus com frete imediato em horário comercial. Para interior do AM e demais regiões do Brasil, "
            "enviamos por Correios (PAC/Sedex). Se me passar o CEP, eu já te digo o prazo e o valor. "
            "Se for Manaus, me diga o bairro que estimo o tempo.")

def tool_ver_produto(termo: str):
    itens = buscar_produto(termo, n=3)
    if not itens:
        return "Não encontrei esse item. Pode me enviar o nome exato ou o SKU?"
    linhas = []
    for p in itens:
        nome    = p.get('nome') or p.get('título') or p.get('titulo') or 'Produto'
        sku     = p.get('sku') or '—'
        preco   = p.get('preco') or p.get('preço') or p.get('valor') or 'sob consulta'
        estoque = p.get('estoque') or p.get('qtd') or p.get('quantidade') or ''
        linhas.append(f"• {nome} (SKU {sku}) – R$ {preco}" + (f" — estoque: {estoque}" if estoque else ""))
    linhas.append("Se quiser, já separo no seu nome. Me diga o SKU ou a opção que gostou.")
    return "\n".join(linhas)

def tool_criar_pedido(state: dict):
    pedido_id = f"DFT-{str(abs(hash(json.dumps(state))) % 100000).zfill(5)}"
    return (f"Pedido rascunho criado: {pedido_id}. Me confirme o método de pagamento (Pix ou cartão em até 6x) "
            f"e o endereço/retirada para finalizar.")

def agente_responder(user_text: str, customer_id: str | None, customer_name: str | None):
    txt = (user_text or "").strip()
    if not txt:
        return None
    db = _load_state()
    st = db.get(customer_id or "anon", {"carrinho": []})

    # a) CEP explícito => cotação
    m = CEP_RE.search(txt)
    if m:
        cep = m.group(0)
        st["cep"] = cep
        db[customer_id or "anon"] = st
        _save_state(db)
        return tool_cotar_frete(cep)

    # b) intenção de entrega/frete sem CEP
    if ENTREGA_RE.search(txt):
        return tool_politica_entrega_sem_cep()

    # c) ver produto (preço/estoque/SKU)
    gatilhos = ["tem ", "estoque", "disponível", "preço", "valor", "sku", "tamanho", "cor"]
    if any(g in txt.lower() for g in gatilhos):
        termo = re.sub(r"\b(tem|de|o|a|um|uma|preço|valor|do|da|no|na|sku|tamanho|cor|disponível|estoque)\b", "", txt, flags=re.I).strip() or txt
        return tool_ver_produto(termo)

    # d) fechamento de pedido
    if re.search(r"\b(finalizar|fechar|comprar|fechar pedido|checkout)\b", txt, flags=re.I):
        return tool_criar_pedido(st)

    return None

# ========= Q&A base (Chroma) =========
def answer_qna(q: str):
    # Se não estiver treinado ou embeddings indisponíveis, retorna None silenciosamente
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection("qna")
        vec = _embed(q)
        if not vec:
            return None
        r = col.query(query_embeddings=[vec], n_results=1, include=["documents","distances"])
        if not r or not r.get("documents") or not r["documents"][0]:
            return None
        doc  = r["documents"][0][0]
        dist = r["distances"][0][0]
        # Threshold mais tolerante p/ variações de frase (ex.: “trabalha com entrega?”)
        if dist <= 0.35:
            return doc.split("RESPOSTA:\n",1)[-1].strip() if "RESPOSTA:" in doc else doc.strip()
        return None
    except Exception:
        return None

# ========= RAG simplificado (opcional) =========
def answer_rag(q: str):
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        col_ap = client.get_or_create_collection("desfrut_apostila")
        col_pd = client.get_or_create_collection("desfrut_produtos")
        vec = _embed(q)
        if not vec: return None
        def _q(col):
            try:
                r = col.query(query_embeddings=[vec], n_results=3, include=["documents"])
                return (r.get("documents") or [[]])[0]
            except Exception:
                return []
        docs = _q(col_ap) + _q(col_pd)
        if not docs:
            return None
        ctx = "\n\n".join(docs[:6])
        if not oai:
            return f"Base consultada. Encontrei {len(docs)} trechos, mas o gerador está fora. Posso te orientar manualmente com produtos/CEP."
        prompt = [
            {"role":"system","content":"Você é a assistente da Desfrut (sexshop em Manaus). Acolhedora, objetiva e educativa. Evite conteúdo explícito."},
            {"role":"user","content":f"Pergunta: {q}\n\nUse o contexto com prioridade:\n{ctx}"}
        ]
        out = oai.chat.completions.create(model=GEN_MODEL, messages=prompt, temperature=0.2)
        return out.choices[0].message.content
    except Exception:
        return None

# ========= Páginas =========
HTML = """
<!doctype html><html><head><meta charset="utf-8"/><title>Desfrut IA</title>
<style>body{font-family:Arial;max-width:820px;margin:40px auto;padding:0 16px}
.box{background:#f8f8f8;padding:16px;border-radius:12px}</style></head>
<body><h1>Desfrut IA</h1><div class="box">
<textarea id="q" rows="3" style="width:100%"></textarea><br/>
<button id="send">Perguntar</button></div><pre id="out"></pre>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})});
  const j=await r.json(); document.getElementById('out').textContent=JSON.stringify(j,null,2);
}
document.getElementById('send').onclick=ask;
</script></body></html>
"""

@app.get("/")
def home(): return render_template_string(HTML)

@app.get("/healthz")
def healthz(): return jsonify(ok=True)

# ========= /ask =========
def _respond(question, cust_id=None, cust_name=None):
    # 1) Q&A (mais tolerante)
    a = answer_qna(question)
    if a: return humanize(a, cust_name)
    # 2) Agente (agora entende “entrega/frete” sem CEP)
    a = agente_responder(question, cust_id, cust_name)
    if a: return humanize(a, cust_name)
    # 3) RAG
    a = answer_rag(question)
    if a: return humanize(a, cust_name)
    # 4) Fallback final
    return humanize("Posso te ajudar com frete (me informe o CEP) ou disponibilidade/preço (me diga o nome/SKU).", cust_name)

@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        q = (data.get("question") or "").strip()
        if not q: return jsonify(error="Pergunta vazia."), 400
        ans = _respond(q, data.get("customer_id"), data.get("customer_name"))
        return jsonify(answer=ans), 200
    except Exception:
        return jsonify(answer=humanize("Tive uma instabilidade, mas já voltei. Pode repetir por favor?")), 200

# ========= Treino via URL e contagem =========
def _train_qna(csv_path: str, reset=False) -> int:
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        if reset:
            try: client.delete_collection("qna")
            except Exception: pass
        col = client.get_or_create_collection("qna")
        docs, metas, ids, vecs = [], [], [], []
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                q = (row.get("pergunta") or "").strip()
                a = (row.get("resposta") or "").strip()
                t = (row.get("tags") or "").strip()
                if not q or not a: continue
                docs.append(q + "\n\nRESPOSTA:\n" + a)
                metas.append({"tags": t})
                ids.append(os.urandom(8).hex())
                vecs.append(_embed(q))
        if not docs: return 0
        try:
            import math
            if all(v is not None for v in vecs):
                col.add(documents=docs, metadatas=metas, ids=ids, embeddings=vecs)
            else:
                col.add(documents=docs, metadatas=metas, ids=ids)
        except Exception:
            col.add(documents=docs, metadatas=metas, ids=ids)
        return len(docs)
    except Exception:
        return 0

@app.get("/admin/train")
def admin_train():
    token = request.args.get("token","")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error="forbidden"), 403
    path  = request.args.get("path","base_qna.csv")
    reset = request.args.get("reset","0").lower() in ("1","true","yes","on")
    try:
        n = _train_qna(path, reset=reset)
        return jsonify(ok=True, inserted=n, reset=reset, path=path)
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=path), 200  # nunca 500

@app.get("/admin/qna_count")
def admin_qna_count():
    token = request.args.get("token","")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error="forbidden"), 403
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection("qna")
        # API 0.5.x tem count()
        try:
            c = col.count()
        except Exception:
            # fallback best-effort
            res = col.get(ids=None, include=[])
            c = len(res.get("ids", []))
        return jsonify(ok=True, count=c)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

# ====== MAIN (dev) ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), debug=False)
