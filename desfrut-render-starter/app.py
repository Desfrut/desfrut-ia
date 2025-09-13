from flask import Flask, request, jsonify, render_template_string
import os, json, re, csv, difflib, random
from datetime import datetime

# ===== Config/env =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEN_MODEL      = os.getenv("GEN_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "/tmp/chroma")
STATE_DB       = os.getenv("STATE_DB", "/tmp/state.json")
PRODUTOS_CSV   = os.getenv("PRODUTOS_CSV", "base_produtos.csv")
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", "")

HUMAN_WHATS    = os.getenv("HUMAN_WHATS", "📲 (92) 9 8424-5930")
HUMAN_HORARIO  = os.getenv("HUMAN_HORARIO", "10h às 22h (Manaus)")
LOJA_ENDERECO  = os.getenv("LOJA_ENDERECO", "Rua Exemplo, 123 — Manaus")

app = Flask(__name__)

# ===== OpenAI opcional =====
oai = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        oai = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        oai = None

# ===== Saudação =====
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

# ===== Microcopy (ajuste à vontade) =====
MICROCOPY = {
    "entrega_manaus_sem_cep": [
        "%%SAUD%%, %%NOME%%! Entregamos em Manaus no mesmo dia 🛵 (10h–22h). Qual é o bairro? Assim estimo o tempo."
    ],
    "entrega_cep_manaus": [
        "Perfeito, %%NOME%%! CEP de Manaus identificado. A entrega é no mesmo dia (10h–22h). Qual o bairro para estimar o tempo?"
    ],
    "entrega_cep_brasil": [
        "Anotado! Para esse CEP envio por Correios (PAC/Sedex). Prefere rapidez (Sedex) ou economia (PAC)?"
    ],
    "entrega_bairro_follow": [
        "Bairro **%%BAIRRO%%**: normalmente 45min a 1h após o fechamento (10h–22h). Quer que eu acione o motoboy?"
    ],
    "frete_politica_sem_cep": [
        "%%SAUD%%, %%NOME%%! Em Manaus entregamos hoje; outras cidades, via Correios (PAC/Sedex). Se me passar o CEP, já calculo prazo e valor."
    ],
    "frete_taxa_manaus": [
        "Em Manaus a entrega é **grátis** em horário comercial. Qual o bairro?"
    ],
    "frete_taxa_brasil": [
        "Fora de Manaus, o valor depende do CEP (PAC/Sedex). Me envia o CEP que eu calculo rapidinho."
    ],
    "produto_lista": [
        "Separei opções:\n%%LINHAS%%\nPosso reservar no seu nome. Qual você prefere?"
    ],
    "pedido_criado": [
        "Pedido rascunho **%%ID%%** criado. Preferir Pix, dinheiro ou cartão (débito/crédito em até 6x)? É **entrega** ou **retirada**?"
    ],
    "pedido_resumo_final": [
        "Fechado, %%NOME%%! Pedido **%%ID%%**:\n• Pagamento: **%%PAG%%**\n• Modalidade: **%%MODAL%%**\n%%ENDERECO%%\nJá vou separar e te atualizo o horário 👌"
    ],
    "handoff_humano": [
        "Claro, conecto com uma especialista agora ❤️  Contato: %%WHATS%% — atendimento %%HORARIO%%."
    ],
    "address_pedido": [
        "Me envia o **endereço completo** (rua, número e ponto de referência)."
    ],
    "ask_delivery": [
        "Vai querer **entrega** ou **retirada**?"
    ],
    "ask_payment": [
        "Quer pagar por **Pix**, **dinheiro** ou **cartão** (débito/crédito até 6x)?"
    ],
    "fallback": [
        "Posso te ajudar com frete (me mande o CEP) ou disponibilidade/preço (me diga o nome/SKU)."
    ]
}

def _mc(key, **kws):
    t = random.choice(MICROCOPY[key])
    t = t.replace("%%SAUD%%", _saudacao())
    t = t.replace("%%NOME%%", (kws.get("nome","") or "").split(" ")[0])
    t = t.replace("%%BAIRRO%%", kws.get("bairro","") or "")
    t = t.replace("%%ID%%", kws.get("pedido_id","") or "")
    t = t.replace("%%PAG%%", kws.get("pag","") or "")
    t = t.replace("%%MODAL%%", kws.get("modal","") or "")
    t = t.replace("%%WHATS%%", HUMAN_WHATS)
    t = t.replace("%%HORARIO%%", HUMAN_HORARIO)
    t = t.replace("%%ENDERECO%%", kws.get("endereco","").strip())
    return t

def humanize(texto: str, nome: str | None = None) -> str:
    primeiros = (texto or "")[:20].lower()
    if any(primeiros.startswith(x) for x in ["oi", "olá", "ola", "bom ", "boa "]):
        return texto
    pref = _saudacao()
    if nome:
        return f"{pref}, {nome.split(' ')[0]}! {texto}"
    return f"{pref}! {texto}"

# ===== Embeddings (silenciosos) =====
def _embed(text: str):
    if not oai: return None
    try:
        e = oai.embeddings.create(model=EMBED_MODEL, input=text)
        return e.data[0].embedding
    except Exception:
        return None

# ===== Produtos =====
def _carregar_produtos():
    itens = []
    try:
        with open(PRODUTOS_CSV, newline='', encoding='utf-8') as f:
            for r in csv.DictWriter(f, fieldnames=[]):  # no-op se não for usado
                pass
    except Exception:
        pass
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

# ===== Memória =====
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
def _salvar_contexto(db, st_key, **kwargs):
    st = db.get(st_key, {"carrinho": []})
    st.update({k: v for k, v in kwargs.items() if v is not None})
    db[st_key] = st
    _save_state(db)
    return st

# ===== Intenções/regex =====
CEP_RE      = re.compile(r"\b\d{5}-?\d{3}\b")
ENTREGA_RE  = re.compile(r"\b(entrega|entregam|entregas|frete|enviar|envia|delivery|motoboy|retirada|retira|buscar)\b", re.I)
TAXA_RE     = re.compile(r"\b(taxa|taxas|cobra|cobram|cobrar|cobrança|custa|valor da entrega|taxa de entrega)\b", re.I)
HUMANO_RE   = re.compile(r"\b(humano|atendente|vendedor(a)?|pessoa|falar com humano|suporte|telefone|n(ú|u)mero|whats|whatsapp)\b", re.I)
BAIRRO_CONFIRM_RE = re.compile(r"\b(informei (acima|antes)|ja? informei|mesmo bairro|o mesmo|como disse)\b", re.I)
FINALIZE_RE = re.compile(r"\b(finaliza(r)?|pode (enviar|acionar|chamar)|segue? (com|a) entrega|pode fechar|vamos fechar|quero (fechar|finalize)|fechar pedido|checkout)\b", re.I)
PAYMENT_RE  = re.compile(r"\b(pix|dinheiro|d(é|e)b(í|i)to|cr(é|e)dito|cart(ã|a)o|6x|6 parcelas|parcelar)\b", re.I)
RETIRADA_RE = re.compile(r"\b(retira(da)?|pegar na loja|buscar na loja|retirar na loja)\b", re.I)
ENTREGA_CHOICE_RE = re.compile(r"\b(entrega|motoboy|enviar)\b", re.I)
ADDRESS_HINT_RE = re.compile(r"\b(rua|avenida|av\.?|travessa|condom(í|i)nio|residencial|bloco|casa|apto|apartamento|n(º|o)|numero|número)\b", re.I)

# NOVO: confirmações curtas
YES_RE = re.compile(r"\b(sim|ok|pode|manda|bora|claro|confirmo|isso|isso mesmo|perfeito|fechou)\b", re.I)
NO_RE  = re.compile(r"\b(n(a|ã)o|nao|melhor n(a|ã)o|negativo|deixa)\b", re.I)

# ===== Ferramentas =====
def tool_cotar_frete(cep: str, nome=None):
    cep = re.sub(r"\D", "", cep)
    if cep.startswith("690"):
        return _mc("entrega_cep_manaus", nome=nome)
    return _mc("entrega_cep_brasil", nome=nome)

def tool_taxa_resposta(intent_frete: bool, bairro: str | None, nome=None):
    if bairro:
        return humanize("Dentro de Manaus a entrega é **grátis** em horário comercial. Posso acionar o motoboy?", nome)
    if intent_frete:
        return _mc("frete_taxa_manaus", nome=nome)
    return humanize("Em Manaus a entrega é grátis em horário comercial; para outras cidades, o valor depende do CEP (PAC/Sedex). Me manda o CEP que eu calculo já.", nome)

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
    return _mc("produto_lista", linhas="\n".join(linhas))

def _novo_pedido_id(st):
    return f"DFT-{str(abs(hash(json.dumps(st))) % 100000).zfill(5)}"

def tool_criar_pedido(st, nome=None):
    pedido_id = _novo_pedido_id(st)
    return _mc("pedido_criado", pedido_id=pedido_id)

def tool_finalizar_pedido(st, nome=None):
    pedido_id = st.get("pedido_id") or _novo_pedido_id(st)
    st["pedido_id"] = pedido_id
    if st.get("modal") == "Retirada":
        end = f"• Retirada: **{LOJA_ENDERECO}**"
    else:
        end = f"• Endereço: **{st.get('endereco','(aguardando)')}**"
    return _mc("pedido_resumo_final", nome=nome, pedido_id=pedido_id,
               pag=st.get("pag","—"), modal=st.get("modal","—"), endereco=end)

# ===== Agente =====
def agente_responder(user_text: str, customer_id: str | None, customer_name: str | None):
    txt = (user_text or "").strip()
    if not txt:
        return None

    st_key = customer_id or "anon"
    db = _load_state()
    st = db.get(st_key, {"carrinho": []})

    # 0) Handoff humano
    if HUMANO_RE.search(txt):
        _salvar_contexto(db, st_key, intent="humano", next=None)
        return _mc("handoff_humano")

    # 1) Se ela acabou de oferecer acionar motoboy, aceite “sim/ok/finaliza”
    if st.get("next") == "confirm_finalize":
        if YES_RE.search(txt) or FINALIZE_RE.search(txt):
            st = _salvar_contexto(db, st_key, intent="checkout", next=None,
                                  pedido_id=st.get("pedido_id") or _novo_pedido_id(st))
            return tool_criar_pedido(st)
        if NO_RE.search(txt):
            _salvar_contexto(db, st_key, next=None)
            return humanize("Sem problemas 🙂. Se preferir, posso te ajudar com outro produto ou calcular frete por CEP.", customer_name)
        # não respondeu sim/não → segue fluxo normal abaixo (sem virar bairro)

    # 2) Finalizar/acionar motoboy (comandos diretos)
    if FINALIZE_RE.search(txt):
        st = _salvar_contexto(db, st_key, intent="checkout", next=None,
                              pedido_id=st.get("pedido_id") or _novo_pedido_id(st))
        return tool_criar_pedido(st)

    # 3) CEP explícito
    m = CEP_RE.search(txt)
    if m:
        cep = m.group(0)
        st = _salvar_contexto(db, st_key, cep=cep, intent="frete", next=None)
        return tool_cotar_frete(cep, nome=customer_name)

    # 4) “já informei” / “mesmo bairro”
    if BAIRRO_CONFIRM_RE.search(txt):
        bairro = st.get("bairro")
        if bairro:
            _salvar_contexto(db, st_key, next="confirm_finalize")
            return _mc("entrega_bairro_follow", bairro=bairro)
        _salvar_contexto(db, st_key, intent="frete", next=None)
        return humanize("Consegue me dizer o bairro, por favor? Assim eu te passo a janela certinha.", customer_name)

    # 5) intenção de entrega/frete sem CEP
    if ENTREGA_RE.search(txt):
        st = _salvar_contexto(db, st_key, intent="frete")
        if st.get("bairro"):
            _salvar_contexto(db, st_key, next="confirm_finalize")
            return _mc("entrega_bairro_follow", bairro=st["bairro"])
        return _mc("entrega_manaus_sem_cep", nome=customer_name)

    # 6) “tem taxa?” / “cobra entrega?”
    if TAXA_RE.search(txt):
        intent_frete = st.get("intent") == "frete"
        bairro = st.get("bairro")
        return tool_taxa_resposta(intent_frete, bairro, nome=customer_name)

    # 7) Se estamos em frete e a mensagem parece um bairro curto → salvar bairro
    if st.get("intent") == "frete" and not CEP_RE.search(txt):
        bairro_txt = re.sub(r"^(sou do|sou da|sou de|bairro|do bairro|da|de)\s+", "", txt, flags=re.I).strip()
        if 1 <= len(bairro_txt) <= 40 and \
           not ENTREGA_RE.search(bairro_txt) and \
           not TAXA_RE.search(bairro_txt) and \
           not BAIRRO_CONFIRM_RE.search(bairro_txt) and \
           not YES_RE.search(bairro_txt) and not NO_RE.search(bairro_txt) and \
           not FINALIZE_RE.search(bairro_txt):
            st = _salvar_contexto(db, st_key, bairro=bairro_txt, next="confirm_finalize")
            return _mc("entrega_bairro_follow", bairro=bairro_txt)

    # 8) Checkout: pagamento / entrega-retirada / endereço
    if st.get("intent") == "checkout":
        # pagamento
        if PAYMENT_RE.search(txt):
            pag = "Pix" if re.search(r"pix", txt, re.I) else \
                  "Dinheiro" if re.search(r"dinheiro", txt, re.I) else "Cartão (débito/crédito até 6x)"
            st = _salvar_contexto(db, st_key, pag=pag)
            if st.get("modal") == "Retirada":
                return tool_finalizar_pedido(st, nome=customer_name)
            if st.get("modal") == "Entrega":
                if st.get("endereco"):
                    return tool_finalizar_pedido(st, nome=customer_name)
                return humanize(random.choice(MICROCOPY["address_pedido"]), customer_name)
            return humanize(random.choice(MICROCOPY["ask_delivery"]), customer_name)

        # modalidade: retirada
        if RETIRADA_RE.search(txt):
            st = _salvar_contexto(db, st_key, modal="Retirada", endereco=LOJA_ENDERECO)
            if st.get("pag"):
                return tool_finalizar_pedido(st, nome=customer_name)
            return humanize(random.choice(MICROCOPY["ask_payment"]), customer_name)

        # modalidade: entrega
        if ENTREGA_CHOICE_RE.search(txt):
            st = _salvar_contexto(db, st_key, modal="Entrega")
            if st.get("endereco") and st.get("pag"):
                return tool_finalizar_pedido(st, nome=customer_name)
            if not st.get("endereco"):
                return humanize(random.choice(MICROCOPY["address_pedido"]), customer_name)
            if not st.get("pag"):
                return humanize(random.choice(MICROCOPY["ask_payment"]), customer_name)

        # endereço livre
        if ADDRESS_HINT_RE.search(txt) and len(txt) > 8:
            st = _salvar_contexto(db, st_key, endereco=txt, modal=st.get("modal") or "Entrega")
            if st.get("pag"):
                return tool_finalizar_pedido(st, nome=customer_name)
            return humanize(random.choice(MICROCOPY["ask_payment"]), customer_name)

        # lembrete do que falta
        faltantes = []
        if not st.get("pag"): faltantes.append("pagamento")
        if not st.get("modal"): faltantes.append("entrega ou retirada")
        if st.get("modal") == "Entrega" and not st.get("endereco"): faltantes.append("endereço")
        if faltantes:
            dica = " / ".join(faltantes)
            return humanize(f"Me diga por favor: {dica}.", customer_name)

    # 9) Ver produto (preço/estoque/SKU)
    gatilhos = ["tem ", "estoque", "disponível", "preço", "valor", "sku", "tamanho", "cor"]
    if any(g in txt.lower() for g in gatilhos):
        termo = re.sub(r"\b(tem|de|o|a|um|uma|preço|valor|do|da|no|na|sku|tamanho|cor|disponível|estoque)\b", "", txt, flags=re.I).strip() or txt
        return tool_ver_produto(termo)

    # 10) Nada casou → deixa Q&A/RAG
    return None

# ===== Q&A =====
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

# ===== RAG (opcional) =====
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

# ===== Páginas e /ask =====
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

def _respond(question, cust_id=None, cust_name=None):
    a = answer_qna(question)
    if a: return humanize(a, cust_name)
    a = agente_responder(question, cust_id, cust_name)
    if a: return a
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

# ===== Treino via URL =====
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), debug=False)
