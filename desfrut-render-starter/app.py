# app.py — Desfrut IA (humanizada v2: taxa de entrega + handoff humano)
# - Intenções: entrega (com/sem CEP), taxa de entrega, produtos, fechar pedido, falar com humano
# - Follow-up: lembra que a conversa é sobre frete e bairro
# - Q&A + RAG + /admin/train + /admin/qna_count
# - Resiliente: /ask nunca 500

from flask import Flask, request, jsonify, render_template_string
import os, json, re, csv, difflib, random
from datetime import datetime

# ===== Config/env =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEN_MODEL      = os.getenv("GEN_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "/tmp/chroma")      # mude p/ /data/chroma se tiver Disk
STATE_DB       = os.getenv("STATE_DB", "/tmp/state.json")    # mude p/ /data/state.json se tiver Disk
PRODUTOS_CSV   = os.getenv("PRODUTOS_CSV", "base_produtos.csv")
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", "")

# Contatos para handoff (ajuste nas ENV do Render)
HUMAN_WHATS    = os.getenv("HUMAN_WHATS", "📲 (92) 9 8424-5930")
HUMAN_HORARIO  = os.getenv("HUMAN_HORARIO", "10h às 22h (Manaus)")

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
        if 5 <= h < 12:  return "Bom dia"
        if 12 <= h < 18: return "Boa tarde"
        return "Boa noite"
    except Exception:
        return "Oi"

# ===== “Voz” com variações =====
MICROCOPY = {
    "entrega_manaus_sem_cep": [
        "%%SAUD%%, %%NOME%%! Entregamos em Manaus no mesmo dia 🛵, entre 45min a 1h, nosso Delivery funciona das 9h as 22h. Qual é o seu bairro?",
        "%%SAUD%%, %%NOME%%! Sim, fazemos entrega hoje de imediado e sem taxa para toda Manaus (10h–22h). Qual bairro? Assim estimamos o tempo."
    ],
    "entrega_cep_manaus": [
        "Perfeito, %%NOME%%! CEP de Manaus identificado. A entrega é no mesmo dia, entre 45min a 1h, nosso Delivery funciona das 9h as 22h. Me diz o bairro que estimo o tempo 😉",
        "Show! Em Manaus a gente entrega hoje mesmo, de imediado e sem taxa. Qual o bairro pra eu estimar?"
    ],
    "entrega_cep_brasil": [
        "Anotado, %%NOME%%! Para esse CEP, envio por Correios (PAC/Sedex). Prefere rapidez (Sedex) ou economia (PAC)?",
        "Beleza! Pra esse CEP faço PAC ou Sedex — você prefere mais rápido ou mais econômico?"
    ],
    "entrega_bairro_follow": [
        "Bairro **%%BAIRRO%%**: normalmente 45min a 1h após o fechamento do pedido (10h–22h). Posso acionar o motoboy?",
        "Em **%%BAIRRO%%**, a janela costuma ser de 45min a 1h (10h–22h). Quer que eu finalize com a entrega?"
    ],
    "frete_politica_sem_cep": [
        "%%SAUD%%, %%NOME%%! Em Manaus entregamos no mesmo dia. Interior via Barco ou Ônibus. Demais cidades: Correios (PAC/Sedex). Se me passar o CEP, calculo prazo e valor.",
        "%%SAUD%%, %%NOME%%! Manaus: entrega hoje mesmo; Interior via Barco ou Ônibus, outras regiões: PAC/Sedex. Me manda o CEP que eu cotar rapidinho."
    ],
    "frete_taxa_manaus": [
        "Em Manaus, a entrega é no mesmo dia, entre 45min a 1h, nosso Delivery funciona das 9h as 22h e é **grátis**. Me diz o bairro que eu te passo a janela 😉",
        "Dentro de Manaus a entrega sai **sem taxa** (10h–22h). Qual é o bairro?"
    ],
    "frete_taxa_brasil": [
        "Fora de Manaus, o valor depende do **PAC/Sedex** e do CEP. Se me enviar o CEP, eu já te informo o valor certinho.",
        "Para o Interior do Amazonas, cobramos a taxa de embarque do barco de R$15,00; Se for por Ônibus o frete varia entre R$30 a R$60,00; Para outras cidades, a taxa varia por Correios (PAC/Sedex). Me envia o CEP que eu calculo."
    ],
    "produto_lista": [
        "Separei opções:\n%%LINHAS%%\nSe quiser, já deixo reservado no seu nome. Qual você prefere?",
        "Olha o que encontrei:\n%%LINHAS%%\nMe fala o SKU ou a opção que você curtiu e eu já separo."
    ],
    "pedido_criado": [
        "Pedido rascunho **%%ID%%** criado. Preferir Dinheiro, Pix, débito ou cartão (até 6x)? Vai querer entrega ou retirada?",
        "Anotei o rascunho **%%ID%%**. Vamos de Dinheiro, Pix, débito ou cartão (até 6x)? É pra entrega ou retirada?"
    ],
    "handoff_humano": [
        "Claro, conecto você com uma Especialista agora ❤️  Contato: %%WHATS%% — atendimento %%HORARIO%%.",
        "Sem problemas! Te passo com uma Especialista 🙌  Fale no %%WHATS%% (%%HORARIO%%)."
    ],
    "fallback": [
        "Posso te ajudar com frete (me manda o CEP) ou disponibilidade/preço (me diga o nome/SKU).",
        "Se preferir, já vejo frete (CEP) ou checo um produto específico (nome/SKU)."
    ]
}

def _mc(key, nome=None, bairro=None, pedido_id=None, linhas=None):
    t = random.choice(MICROCOPY[key])
    t = t.replace("%%SAUD%%", _saudacao())
    t = t.replace("%%NOME%%", (nome.split(" ")[0] if nome else "").strip())
    t = t.replace("%%WHATS%%", HUMAN_WHATS)
    t = t.replace("%%HORARIO%%", HUMAN_HORARIO)
    if bairro:    t = t.replace("%%BAIRRO%%", bairro.strip())
    if pedido_id: t = t.replace("%%ID%%", pedido_id)
    if linhas:    t = t.replace("%%LINHAS%%", linhas.strip())
    return t

def humanize(texto: str, nome: str | None = None) -> str:
    primeiros = (texto or "")[:20].lower()
    if any(primeiros.startswith(x) for x in ["oi", "olá", "ola", "bom ", "boa "]):
        return texto
    pref = _saudacao()
    if nome:
        return f"{pref}, {nome.split(' ')[0]}! {texto}"
    return f"{pref}! {texto}"

# ========= Embeddings (silenciosos) =========
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

# Detectores de intenção (entrega, taxa, humano, “use o bairro que já falei”, finalizar)
CEP_RE      = re.compile(r"\b\d{5}-?\d{3}\b")
ENTREGA_RE  = re.compile(r"\b(entrega|entregam|entregas|frete|envia|enviam|delivery|motoboy|retirada|retira|buscar)\b", re.I)
TAXA_RE     = re.compile(r"\b(taxa|taxas|cobra|cobram|cobrar|cobrança|custa|valor da entrega|taxa de entrega)\b", re.I)
HUMANO_RE   = re.compile(r"\b(humano|atendente|vendedor(a)?|pessoa|falar com alguém|falar com humano|suporte|telefone|número|numero|whats|whatsapp)\b", re.I)
BAIRRO_CONFIRM_RE = re.compile(r"\b(informei (acima|antes)|já (informei|falei)|mesmo bairro|o mesmo|é esse|como disse)\b", re.I)
FINALIZE_RE = re.compile(r"\b(finaliza(r)?|pode (enviar|acionar|chamar)|segue? (com|a) entrega|pode fechar|vamos fechar|quero que finalize|fechar pedido)\b", re.I)

def tool_cotar_frete(cep: str, nome=None):
    cep = re.sub(r"\D", "", cep)
    if cep.startswith("690"):
        return _mc("entrega_cep_manaus", nome=nome)
    return _mc("entrega_cep_brasil", nome=nome)

def tool_politica_entrega_sem_cep(nome=None):
    return _mc("frete_politica_sem_cep", nome=nome)

def tool_taxa_resposta(intent_frete: bool, bairro: str | None, nome=None):
    if bairro:
        # Se já sabemos que é Manaus (bairro informado), seja direto:
        return humanize("Dentro de Manaus a entrega é **grátis** entre 45min a 1h apos o fechamento do pedido. Deseja finalizar seu pedido?", nome)
    if intent_frete:
        # Estamos falando de frete, mas sem bairro/CEP ainda
        return _mc("frete_taxa_manaus", nome=nome)
    # Usuário falou de taxa sem estarmos em frete: dê visão geral
    return humanize("Em Manaus a entrega é grátis de imediato, discreta, entre 45min a 1h após o fechamento do pedido; para o Interior do AM enviamos via Barco ou Ônibus, outras cidades, o valor depende do CEP (PAC/Sedex). Me manda o CEP que eu calculo rapidinho.", nome)

def tool_ver_produto(termo: str, nome=None):
    itens = buscar_produto(termo, n=3)
    if not itens:
        return humanize("Não encontrei esse item. Pode me enviar o nome exato ou o SKU?", nome)
    linhas = []
    for p in itens:
        nomep   = p.get('nome') or p.get('título') or p.get('titulo') or 'Produto'
        sku     = p.get('sku') or '—'
        preco   = p.get('preco') or p.get('preço') or p.get('valor') or 'sob consulta'
        estoque = p.get('estoque') or p.get('qtd') or p.get('quantidade') or ''
        linha = f"• {nomep} (SKU {sku}) – R$ {preco}" + (f" — estoque: {estoque}" if estoque else "")
        linhas.append(linha)
    return _mc("produto_lista", nome=nome, linhas="\n".join(linhas))

def tool_criar_pedido(state: dict, nome=None):
    pedido_id = f"DFT-{str(abs(hash(json.dumps(state))) % 100000).zfill(5)}"
    return _mc("pedido_criado", nome=nome, pedido_id=pedido_id)

def tool_handoff_humano(nome=None):
    return _mc("handoff_humano", nome=nome)

def _salvar_contexto(db, st_key, **kwargs):
    st = db.get(st_key, {"carrinho": []})
    st.update({k:v for k,v in kwargs.items() if v is not None})
    db[st_key] = st
    _save_state(db)
    return st

def agente_responder(user_text: str, customer_id: str | None, customer_name: str | None):
    txt = (user_text or "").strip()
    if not txt:
        return None

    st_key = customer_id or "anon"
    db = _load_state()
    st = db.get(st_key, {"carrinho": []})

    # 0) Handoff humano
    if HUMANO_RE.search(txt):
        _salvar_contexto(db, st_key, intent="humano")
        return tool_handoff_humano(customer_name)

    # a) CEP explícito => cotação
    m = CEP_RE.search(txt)
    if m:
        cep = m.group(0)
        st = _salvar_contexto(db, st_key, cep=cep, intent="frete")
        return tool_cotar_frete(cep, nome=customer_name)

    # b) intenção de entrega/frete sem CEP
    if ENTREGA_RE.search(txt):
        st = _salvar_contexto(db, st_key, intent="frete")
        return _mc("entrega_manaus_sem_cep", nome=customer_name)

    # c) “tem taxa?” / “cobra entrega?”
    if TAXA_RE.search(txt):
        intent_frete = st.get("intent") == "frete"
        bairro = st.get("bairro")
        return tool_taxa_resposta(intent_frete, bairro, nome=customer_name)

    # d) se já estamos em frete e a mensagem é curta, trate como bairro
    if st.get("intent") == "frete" and len(txt) <= 60 and not CEP_RE.search(txt):
        st = _salvar_contexto(db, st_key, bairro=txt)
        return _mc("entrega_bairro_follow", nome=customer_name, bairro=txt)

    # e) ver produto (preço/estoque/SKU)
    gatilhos = ["tem ", "estoque", "disponível", "preço", "valor", "sku", "tamanho", "cor"]
    if any(g in txt.lower() for g in gatilhos):
        termo = re.sub(r"\b(tem|de|o|a|um|uma|preço|valor|do|da|no|na|sku|tamanho|cor|disponível|estoque)\b", "", txt, flags=re.I).strip() or txt
        return tool_ver_produto(termo, nome=customer_name)

    # f) fechamento de pedido
    if re.search(r"\b(finalizar|fechar|comprar|fechar pedido|checkout)\b", txt, flags=re.I):
        return tool_criar_pedido(st, nome=customer_name)

    return None

# ========= Q&A (Chroma) =========
def answer_qna(q: str):
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
        if dist <= 0.35:
            return doc.split("RESPOSTA:\n",1)[-1].strip() if "RESPOSTA:" in doc else doc.strip()
        return None
    except Exception:
        return None

# ========= RAG (opcional) =========
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
            return "Consultei a base, mas o gerador está instável agora. Posso te orientar manualmente com produtos/CEP."
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
    a = answer_qna(question)
    if a: return humanize(a, cust_name)
    a = agente_responder(question, cust_id, cust_name)
    if a: return a  # já vem humanizada pelos templates quando for o caso
    a = answer_rag(question)
    if a: return humanize(a, cust_name)
    return humanize(random.choice(MICROCOPY["fallback"]), cust_name)

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
        return jsonify(ok=False, error=str(e), path=path), 200

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
        try:
            c = col.count()
        except Exception:
            res = col.get(ids=None, include=[])
            c = len(res.get("ids", []))
        return jsonify(ok=True, count=c)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

# ====== MAIN (dev) ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), debug=False)
