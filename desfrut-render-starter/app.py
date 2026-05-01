# app.py — Desfrut IA v4.6-safe (único/definitivo)
from flask import Flask, request, jsonify, render_template_string
import os, json, re, csv, difflib, random
from datetime import datetime

# ===== Config =====
STATE_DB       = os.getenv("STATE_DB", "/data/state.json")
PRODUTOS_CSV   = os.getenv("PRODUTOS_CSV", "base_produtos.csv")
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", "desfrut123")
HUMAN_WHATS    = os.getenv("HUMAN_WHATS", "📲 (92) 9 8424-5930")
HUMAN_HORARIO  = os.getenv("HUMAN_HORARIO", "10h às 22h (Manaus)")
LOJA_ENDERECO  = os.getenv("LOJA_ENDERECO", "Rua Exemplo, 123 — Manaus")
AGENT_VERSION  = "4.6-safe"

app = Flask(__name__)

def _saudacao():
    h = datetime.now().hour
    if 5 <= h < 12:  return "Bom dia"
    if 12 <= h < 18: return "Boa tarde"
    return "Boa noite"

def humanize(texto: str, nome: str | None = None) -> str:
    texto = (texto or "").strip()
    if not texto: return "Posso te ajudar com mais alguma coisa?"
    primeiros = texto[:20].lower()
    if any(primeiros.startswith(x) for x in ["oi","olá","ola","bom ","boa "]):
        pref = ""
    else:
        pref = _saudacao() + (f", {nome.split(' ')[0]}" if nome else "") + "! "
    return (pref + texto).strip()

MICROCOPY = {
    "entrega_manaus_sem_cep": ["%%SAUD%%, %%NOME%%! Entregamos em Manaus no mesmo dia (10h–22h). Qual é o bairro? Assim estimo a janela certinha."],
    "entrega_cep_manaus": ["Perfeito! CEP de Manaus identificado. A entrega é no mesmo dia (10h–22h). Qual o bairro para estimar a janela?"],
    "entrega_cep_brasil": ["Para este CEP, envio por Correios (PAC/Sedex). Prefere rapidez (Sedex) ou economia (PAC)?"],
    "entrega_bairro_follow": ["Bairro **%%BAIRRO%%**: normalmente 45–90 min após o fechamento (10h–22h). Quer que eu acione o motoboy?"],
    "produto_lista": ["Separei opções:\n%%LINHAS%%\nQual você quer? Pode me mandar o **SKU**."],
    "pedir_produto": ["Perfeito! Qual **produto** você deseja? Me diga o **nome** ou **SKU** que eu já separo."],
    "pedido_criado": ["Pedido rascunho **%%ID%%** criado. Você prefere **Pix**, **dinheiro** ou **cartão** (débito/crédito até 6x)? É **entrega** ou **retirada**?"],
    "pedido_resumo_final": ["Fechado! Pedido **%%ID%%**:\n• Itens: **%%ITENS%%**\n• Pagamento: **%%PAG%%**\n• Modalidade: **%%MODAL%%**\n%%ENDERECO%%\nJá vou separar e te atualizo o horário 👌"],
    "handoff_humano": ["Sem problema! Te conecto com uma especialista ❤️  Contato: %%WHATS%% — atendimento %%HORARIO%%."],
    "address_pedido": ["Me envia o **endereço completo** (rua, número e ponto de referência)."],
    "ask_delivery": ["Vai querer **entrega** ou **retirada**?"],
    "ask_payment": ["Quer pagar por **Pix**, **dinheiro** ou **cartão** (débito/crédito até 6x)?"],
    "fallback": ["Posso te ajudar com frete (me mande o CEP) ou disponibilidade/preço (me diga o nome/SKU)."],
    "tempo_bairro": ["Para **%%BAIRRO%%**, a janela costuma ser de **45–90 min** após fecharmos o pedido (10h–22h)."]
}

def _mc(key, **kws):
    t = random.choice(MICROCOPY[key])
    repl = {
        "%%SAUD%%": _saudacao(),
        "%%NOME%%": (kws.get("nome","") or "").split(" ")[0],
        "%%BAIRRO%%": kws.get("bairro","") or "",
        "%%ID%%": kws.get("pedido_id","") or "",
        "%%PAG%%": kws.get("pag","") or "",
        "%%MODAL%%": kws.get("modal","") or "",
        "%%ITENS%%": kws.get("itens","") or "",
        "%%WHATS%%": HUMAN_WHATS,
        "%%HORARIO%%": HUMAN_HORARIO,
        "%%ENDERECO%%": (kws.get("endereco","") or "").strip(),
    }
    for k,v in repl.items(): t = t.replace(k, v)
    return t

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
def _st(db, key):
    return db.get(key, {"carrinho": [], "intent": None, "next_step": None, "stage": None})

# ===== Regex =====
CEP_RE      = re.compile(r"\b\d{5}-?\d{3}\b")
ENTREGA_RE  = re.compile(r"\b(entrega|entregam|entregas|frete|enviar|envia|delivery|motoboy|retirada|retira|buscar)\b", re.I)
TAXA_RE     = re.compile(r"\b(taxa|taxas|cobra|cobram|cobrar|cobrança|custa|valor da entrega|taxa de entrega)\b", re.I)
HUMANO_RE   = re.compile(r"\b(humano|atendente|vendedor(a)?|pessoa|falar com humano|suporte|telefone|n(ú|u)mero|whats|whatsapp)\b", re.I)
BAIRRO_CONFIRM_RE = re.compile(r"\b(informei (acima|antes)|ja? informei|mesmo bairro|o mesmo|como disse)\b", re.I)
FINALIZE_RE = re.compile(r"\b(finaliza(r)?|pode (enviar|acionar|chamar)|segue? (com|a) entrega|pode fechar|vamos fechar|quero (fechar|finalize)|fechar pedido|checkout)\b", re.I)
CONFIRM_YES_RE = re.compile(r"^(sim|pode|ok|isso|bora|vamos)$", re.I)
PAYMENT_RE  = re.compile(r"\b(pix|dinheiro|d(é|e)b(í|i)to|cr(é|e)dito|cart(ã|a)o|6x|6 parcelas|parcelar|pagamento)\b", re.I)
RETIRADA_RE = re.compile(r"\b(retira(da)?|pegar na loja|buscar na loja|retirar na loja)\b", re.I)
ENTREGA_CHOICE_RE = re.compile(r"\b(entrega|motoboy|enviar)\b", re.I)
ADDRESS_HINT_RE = re.compile(r"\b(rua|avenida|av\.?|travessa|condom(í|i)nio|residencial|bloco|casa|apto|apartamento|n(º|o)|numero|número)\b", re.I)
EXPLICIT_BAIRRO_RE = re.compile(r"\b(bairro|sou d[eo]|moro (em|na|no)|estou (em|na|no)|no bairro|do bairro|da zona|na zona)\b", re.I)
TEMPO_RE    = re.compile(r"\b(quanto tempo|em quanto tempo|qual o tempo|prazo|demora)\b", re.I)
GREETING_RE = re.compile(r"^(oi|olá|ola|bom dia|boa tarde|boa noite)\b", re.I)
STATUS_RE   = re.compile(r"\b(status|acompanhar|andamento|rastreio|rastreamento|saiu|já saiu|onde est[aá]|prazo de entrega|quando chega)\b", re.I)
RESET_RE    = re.compile(r"\b(cancelar|reset(ar)?|recomeçar|novo pedido|limpar conversa)\b", re.I)

STOPWORDS_SHORT = {"ola","olá","oi","sim","ok","blz","tá","ta","beleza","boa","bom","entrega","entrega hoje"}

# ===== Produtos (CSV + fuzzy + sinônimos) =====
SINONIMOS = {
    "bullet": ["bullet", "vibrador bullet"],
    "vibrador": ["vibrador", "personal", "vibe", "vibrador personal"],
    "kmed": ["k-med", "kmed", "gel k-med", "gel kmed", "k med"],
    "lubrificante": ["lubrificante", "gel íntimo", "gel intimo", "gel"]
}

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
    termo = (termo or "").strip().lower()

    cand_termos = [termo]
    for k, alts in SINONIMOS.items():
        if k in termo:
            cand_termos.extend(alts)
    cand_termos = list(dict.fromkeys(cand_termos))

    # SKU
    for p in PROD_CACHE:
        sku = str(p.get("sku","")).lower()
        if any(ct in sku for ct in cand_termos if len(ct) >= 3):
            return [p]

    # fuzzy por nome
    nomes = [ (p.get('nome') or p.get('título') or p.get('titulo') or '').lower() for p in PROD_CACHE ]
    melhores = set()
    for ct in cand_termos:
        if not ct: continue
        matches = difflib.get_close_matches(ct, nomes, n=n, cutoff=0.4)
        melhores.update(matches)
    res = [p for p in PROD_CACHE if (p.get('nome') or p.get('título') or p.get('titulo') or '').lower() in melhores]

    # fallback por palavra-chave
    if not res and any(x in termo for x in ["bullet","vibrador","kmed","k-med","lubrificante","gel"]):
        for p in PROD_CACHE:
            nomep = (p.get('nome') or p.get('título') or p.get('titulo') or '').lower()
            if any(x in nomep for x in ["bullet","vibrador","k-med","kmed","lubrificante","gel"]):
                res.append(p)
                if len(res) >= n: break
    return res[:n]

def add_item_from_text(st, txt):
    termo = re.sub(r"\b(tem|de|o|a|um|uma|preço|valor|do|da|no|na|sku|tamanho|cor|disponível|estoque)\b", "", txt, flags=re.I).strip() or txt
    itens = buscar_produto(termo, n=3)
    if not itens: return None, "Não encontrei esse item. Pode me enviar o **nome exato** ou o **SKU**?"
    if len(itens) == 1:
        p = itens[0]
        nomep = p.get('nome') or p.get('título') or p.get('titulo') or 'Produto'
        sku   = p.get('sku') or '—'
        st.setdefault("carrinho", []).append({"sku": sku, "nome": nomep})
        return p, f"Adicionei **{nomep} (SKU {sku})** ao carrinho."
    linhas = []
    for p in itens:
        nomep = p.get('nome') or p.get('título') or p.get('titulo') or 'Produto'
        sku   = p.get('sku') or '—'
        preco = p.get('preco') or p.get('preço') or p.get('valor') or 'sob consulta'
        linhas.append(f"• {nomep} (SKU {sku}) – R$ {preco}")
    return None, _mc("produto_lista", linhas="\n".join(linhas))

def tool_cotar_frete(cep: str, nome=None):
    cep = re.sub(r"\D", "", cep)
    if cep.startswith("690"):
        return _mc("entrega_cep_manaus", nome=nome)
    return _mc("entrega_cep_brasil", nome=nome)

def _novo_pedido_id(st): return f"DFT-{str(abs(hash(json.dumps(st))) % 100000).zfill(5)}"

def tool_criar_pedido(st, nome=None):
    if not st.get("pedido_id"):
        st["pedido_id"] = _novo_pedido_id(st)
    return _mc("pedido_criado", pedido_id=st["pedido_id"])

def _itens_txt(st):
    itens = st.get("carrinho") or []
    if not itens: return "(definir)"
    return ", ".join([f'{i.get("nome","Item")} (SKU {i.get("sku","—")})' for i in itens])

def tool_finalizar_pedido(st, nome=None):
    if not st.get("carrinho"):
        return _mc("pedir_produto")
    pedido_id = st.get("pedido_id") or _novo_pedido_id(st)
    st["pedido_id"] = pedido_id
    st["stage"] = "finalizado"
    if st.get("modal") == "Retirada":
        end = f"• Retirada: **{LOJA_ENDERECO}**"
    else:
        end = f"• Endereço: **{st.get('endereco','(aguardando)')}**"
    return _mc("pedido_resumo_final",
               pedido_id=pedido_id,
               itens=_itens_txt(st),
               pag=st.get("pag","—"),
               modal=st.get("modal","—"),
               endereco=end)

def tempo_estimada(st):
    b = st.get("bairro")
    if not b: return None
    return _mc("tempo_bairro", bairro=b)

def looks_like_bairro(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not (1 <= len(t) <= 30): return False
    if re.search(r"[^a-zà-ú\s]", t): return False
    if len(t.split()) > 3: return False
    if t in STOPWORDS_SHORT: return False
    if re.search(r"\b(quero|saber|saiu|tem|faz|fazer|pode|quando|onde|prazo|status)\b", t): return False
    if any(p.search(txt) for p in [ENTREGA_RE, TAXA_RE, HUMANO_RE, FINALIZE_RE, PAYMENT_RE, RETIRADA_RE, TEMPO_RE, GREETING_RE, STATUS_RE, RESET_RE]):
        return False
    return True

def agente_responder(user_text: str, customer_id: str | None, customer_name: str | None):
    txt = (user_text or "").strip()
    if not txt: return None

    st_key = customer_id or "anon"
    db = _load_state()
    st = _st(db, st_key)
    def save(**kw):
        st.update({k:v for k,v in kw.items() if v is not None})
        db[st_key] = st
        _save_state(db)

    if RESET_RE.search(txt):
        db[st_key] = {"carrinho": [], "intent": None, "next_step": None, "stage": None}
        _save_state(db)
        return "Zerei aqui 👌 Podemos recomeçar: quer saber de **entrega** (me diga bairro ou CEP) ou **produto/preço** (me diga o nome ou SKU)?"

    if HUMANO_RE.search(txt):
        save(intent="humano", next_step=None)
        return _mc("handoff_humano")

    if STATUS_RE.search(txt):
        pid = st.get("pedido_id") or "(ainda não criei o número)"
        when = tempo_estimada(st) or "Em geral entregamos em 45–90 min após fechar o pedido (10h–22h)."
        itens = _itens_txt(st)
        return humanize(f"Seu pedido **{pid}** está em andamento. Itens: **{itens}**. {when}", customer_name)

    if st.get("intent") == "checkout":
        if GREETING_RE.search(txt):
            return humanize("Me diga o **produto (nome ou SKU)**. Prefere **Pix**, **dinheiro** ou **cartão**? Será **entrega** ou **retirada**?", customer_name)
        if PAYMENT_RE.search(txt):
            pag = "Pix" if re.search(r"pix", txt, re.I) else ("Dinheiro" if re.search(r"dinheiro", txt, re.I) else "Cartão (débito/crédito até 6x)")
            save(pag=pag)
        if RETIRADA_RE.search(txt): save(modal="Retirada")
        elif ENTREGA_CHOICE_RE.search(txt): save(modal="Entrega")
        if ADDRESS_HINT_RE.search(txt) and len(txt) > 8:
            save(endereco=txt, modal=st.get("modal") or "Entrega")
        if not st.get("carrinho"):
            gat = ["tem ","estoque","disponível","preço","valor","sku","tamanho","cor","lubrificante","kmed","bullet","vibrador"]
            if any(g in txt.lower() for g in gat):
                p, msg = add_item_from_text(st, txt); save()
                if not p: return humanize(msg, customer_name)
        if st.get("modal") == "Retirada" and st.get("pag"):
            return tool_finalizar_pedido(st, customer_name)
        if st.get("modal") == "Entrega" and st.get("pag") and st.get("endereco"):
            return tool_finalizar_pedido(st, customer_name)
        if not st.get("carrinho"): return humanize(_mc("pedir_produto"), customer_name)
        if not st.get("pag"):      return humanize(random.choice(MICROCOPY["ask_payment"]), customer_name)
        if not st.get("modal"):    return humanize(random.choice(MICROCOPY["ask_delivery"]), customer_name)
        if st.get("modal") == "Entrega" and not st.get("endereco"):
            return humanize(random.choice(MICROCOPY["address_pedido"]), customer_name)

    if st.get("next_step") == "acionar" and (CONFIRM_YES_RE.search(txt) or FINALIZE_RE.search(txt)):
        save(intent="checkout", next_step=None, pedido_id=st.get("pedido_id") or _novo_pedido_id(st))
        return humanize(_mc("pedir_produto"), customer_name)

    if FINALIZE_RE.search(txt):
        save(intent="checkout", next_step=None, pedido_id=st.get("pedido_id") or _novo_pedido_id(st))
        return humanize(_mc("pedir_produto"), customer_name)

    m = CEP_RE.search(txt)
    if m:
        cep = m.group(0)
        save(cep=cep, intent="frete", next_step="bairro")
        return tool_cotar_frete(cep, nome=customer_name)

    if (st.get("next_step") == "bairro" or st.get("intent") == "frete") and looks_like_bairro(txt):
        bairro_txt = txt.strip()
        save(intent="frete", bairro=bairro_txt, next_step="acionar")
        return _mc("entrega_bairro_follow", bairro=bairro_txt)

    if EXPLICIT_BAIRRO_RE.search(txt):
        bairro_txt = re.sub(r"^(sou do|sou da|sou de|moro (em|na|no)|estou (em|na|no)|bairro|do bairro|da|de)\s+", "", txt, flags=re.I).strip()
        if looks_like_bairro(bairro_txt):
            save(intent="frete", bairro=bairro_txt, next_step="acionar")
            return _mc("entrega_bairro_follow", bairro=bairro_txt)

    if BAIRRO_CONFIRM_RE.search(txt):
        if st.get("bairro"):
            save(next_step="acionar")
            return _mc("entrega_bairro_follow", bairro=st["bairro"])
        save(intent="frete", next_step="bairro")
        return humanize("Consegue me dizer o bairro, por favor? Assim eu te passo a janela certinha.", customer_name)

    if ENTREGA_RE.search(txt):
        save(intent="frete")
        if st.get("bairro"):
            save(next_step="acionar")
            return _mc("entrega_bairro_follow", bairro=st["bairro"])
        save(next_step="bairro")
        return _mc("entrega_manaus_sem_cep", nome=customer_name)

    if TAXA_RE.search(txt):
        if st.get("bairro"):
            return humanize("Dentro de Manaus a entrega é **grátis** em horário comercial. Posso acionar o motoboy?", customer_name)
        return humanize("Em Manaus a entrega é grátis; para outras cidades, o valor depende do CEP (PAC/Sedex). Me manda o CEP que eu calculo já.", customer_name)

    if TEMPO_RE.search(txt):
        t = tempo_estimada(st)
        if t: return humanize(t, customer_name)
        return humanize("Me diz o bairro e eu te passo a janela certinha 👍", customer_name)

    gat = ["tem ","estoque","disponível","preço","valor","sku","tamanho","cor","lubrificante","kmed","bullet","vibrador"]
    if any(g in txt.lower() for g in gat):
        p, msg = add_item_from_text(st, txt); save()
        return humanize(msg, customer_name)

    return None

# ===== Q&A CSV (sem OpenAI) =====
QNA_CACHE = None
def _load_qna_csv():
    global QNA_CACHE
    if QNA_CACHE is not None: return QNA_CACHE
    QNA_CACHE = []
    try:
        with open("base_qna.csv", newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                per = (r.get("pergunta") or "").strip()
                res = (r.get("resposta") or "").strip()
                if per and res:
                    QNA_CACHE.append((per, res))
    except Exception:
        pass
    return QNA_CACHE

def answer_qna_lexical(q: str):
    rows = _load_qna_csv()
    if not rows: return None
    qlow = q.lower()
    for per,res in rows:
        if per.lower() in qlow or qlow in per.lower():
            return res
    perguntas = [per for per,_ in rows]
    match = difflib.get_close_matches(q, perguntas, n=1, cutoff=0.55)
    if match:
        for per,res in rows:
            if per == match[0]: return res
    return None

# ===== HTML & Admin =====
HTML = """
<!doctype html><html><head><meta charset="utf-8"/><title>Desfrut IA</title>

<style>
body{font-family:Arial;max-width:820px;margin:40px auto;padding:0 16px}
.box{background:#f8f8f8;padding:16px;border-radius:12px}</style></head>
<body><h1>IA Desfrut</h1><div class="box">

<textarea id="q" rows="3" style="width:100%"></textarea><br/>
<button id="send">Perguntar</button></div><pre id="out"></pre>

<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;

  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})});

  const j=await r.json(); document.getElementById('out').textContent = j.answer;

  document.getElementById('q').value = "";
}
document.getElementById('send').onclick=ask;

document.getElementById('q').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});
</script></body></html>
"""
@app.get("/")
def home(): return render_template_string(HTML)

@app.get("/healthz")
def healthz(): return jsonify(ok=True)

@app.get("/admin/version")
def version():
    token = request.args.get("token","")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error="forbidden"), 403
    return jsonify(ok=True, version=AGENT_VERSION)

@app.get("/admin/reset_state")
def admin_reset():
    token = request.args.get("token","")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error="forbidden"), 403
    try:
        if os.path.exists(STATE_DB): os.remove(STATE_DB)
    except Exception: pass
    return jsonify(ok=True, cleared=True)

@app.get("/admin/debug")
def admin_debug():
    token = request.args.get("token","")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error="forbidden"), 403
    qna_count = 0
    try:
        with open("base_qna.csv", newline='', encoding='utf-8') as f:
            for _ in csv.DictReader(f): qna_count += 1
    except Exception:
        qna_count = 0
    global PROD_CACHE
    if PROD_CACHE is None: PROD_CACHE = _carregar_produtos()
    prod_count = len(PROD_CACHE or [])
    sample = []
    for p in (PROD_CACHE or [])[:3]:
        sample.append(p.get("nome") or p.get("título") or p.get("titulo") or "")
    return jsonify(
        ok=True, version=AGENT_VERSION,
        env={"STATE_DB": STATE_DB, "PRODUTOS_CSV": PRODUTOS_CSV},
        qna_count=qna_count,
        prod_count=prod_count, prod_sample=sample,
        prod_file_exists=os.path.exists(PRODUTOS_CSV)
    )

@app.get("/admin/prod_count")
def admin_prod_count():
    token = request.args.get("token","")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify(ok=False, error="forbidden"), 403
    global PROD_CACHE
    if PROD_CACHE is None: PROD_CACHE = _carregar_produtos()
    sample = []
    for p in (PROD_CACHE or [])[:5]:
        sample.append(p.get("nome") or p.get("título") or p.get("titulo") or "")
    return jsonify(ok=True, count=len(PROD_CACHE or []), sample=sample, file=PRODUTOS_CSV)

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    cust_id = data.get("customer_id")
    cust_name = data.get("customer_name")
    if not q: return jsonify(error="Pergunta vazia."), 400

    # 1) Q&A CSV
    a = answer_qna_lexical(q)
    if a: return jsonify(answer=humanize(a, cust_name), source="qna_csv"), 200

    # 2) Agente (frete/bairro/checkout/produtos)
    a = agente_responder(q, cust_id, cust_name)
    if a: return jsonify(answer=a, source="agent"), 200

    # 3) Fallback
    return jsonify(answer=humanize(random.choice(MICROCOPY["fallback"]), cust_name), source="fallback"), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), debug=False)
