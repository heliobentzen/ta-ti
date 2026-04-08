# Prática 09 — Pipeline de IA Completo

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 09](../conteudo/parte-09-ferramentas-com-ia.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Construir uma API REST com FastAPI que expõe funcionalidades de IA
- Criar uma interface web simples com Streamlit
- Implementar processamento assíncrono de tarefas de IA
- Adicionar observabilidade (logs, métricas) a um pipeline de IA

---

## 🔧 Setup

```bash
pip install openai fastapi uvicorn streamlit python-dotenv chromadb sentence-transformers
```

---

## 📝 Exercício 1 — API REST com FastAPI

Crie `pratica09/api/main.py`:

```python
# pratica09/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import time
import uuid
from datetime import datetime
from typing import Optional

load_dotenv()

app = FastAPI(
    title="IA API — IFPE TA-TI",
    description="API de demonstração com funcionalidades de IA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

# ── Armazenamento em memória (em produção use BD) ──
jobs = {}          # processamento assíncrono
request_log = []   # log de requisições

# ────────────────────────────────────
# MODELOS PYDANTIC
# ────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    system_prompt: Optional[str] = "Você é um assistente útil."
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(500, ge=50, le=4000)

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    latency_ms: float
    request_id: str

class ClassifyRequest(BaseModel):
    text: str
    categories: list[str]

class SummarizeRequest(BaseModel):
    text: str
    max_words: int = 100
    style: str = "neutro"  # neutro, formal, casual

class AsyncJobRequest(BaseModel):
    texts: list[str]
    operation: str  # "summarize" | "classify_sentiment" | "extract_keywords"

# ────────────────────────────────────
# MIDDLEWARE DE LOGGING
# ────────────────────────────────────

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    
    request_log.append({
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": str(request.url.path),
        "status": response.status_code,
        "duration_ms": round(duration, 2)
    })
    
    return response

# ────────────────────────────────────
# ENDPOINTS
# ────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0", "total_requests": len(request_log)}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Endpoint principal de chat com LLM."""
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.message}
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens,
            latency_ms=round((time.time() - start) * 1000, 2),
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify(req: ClassifyRequest):
    """Classifica texto em categorias definidas pelo usuário."""
    categories_str = ", ".join(req.categories)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classifique o texto em UMA das categorias: {categories_str}
            Retorne apenas o nome da categoria, sem explicações.
            
            Texto: {req.text}"""
        }],
        temperature=0,
        max_tokens=50
    )
    
    return {
        "text": req.text[:100],
        "category": response.choices[0].message.content.strip(),
        "available_categories": req.categories
    }

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    """Sumariza um texto."""
    style_instructions = {
        "neutro": "de forma objetiva",
        "formal": "em linguagem formal e técnica",
        "casual": "de forma simples e informal"
    }.get(req.style, "de forma objetiva")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Resuma o texto abaixo {style_instructions} em no máximo {req.max_words} palavras.
            
            {req.text}"""
        }],
        temperature=0.3
    )
    
    summary = response.choices[0].message.content
    return {
        "original_length": len(req.text.split()),
        "summary_length": len(summary.split()),
        "summary": summary
    }

@app.post("/jobs/start")
def start_async_job(req: AsyncJobRequest, background_tasks: BackgroundTasks):
    """Inicia processamento assíncrono de múltiplos textos."""
    job_id = str(uuid.uuid4())[:12]
    
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "operation": req.operation,
        "total": len(req.texts),
        "processed": 0,
        "results": [],
        "started_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(process_job, job_id, req.texts, req.operation)
    
    return {"job_id": job_id, "status": "started", "total_texts": len(req.texts)}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Verifica status de um job assíncrono."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return job

@app.get("/metrics")
def metrics():
    """Métricas de uso da API."""
    if not request_log:
        return {"total_requests": 0}
    
    avg_latency = sum(r["duration_ms"] for r in request_log) / len(request_log)
    status_counts = {}
    for r in request_log:
        status_counts[str(r["status"])] = status_counts.get(str(r["status"]), 0) + 1
    
    return {
        "total_requests": len(request_log),
        "avg_latency_ms": round(avg_latency, 2),
        "status_distribution": status_counts,
        "recent_requests": request_log[-5:]
    }

# ────────────────────────────────────
# BACKGROUND TASKS
# ────────────────────────────────────

def process_job(job_id: str, texts: list, operation: str):
    """Processa textos em background."""
    jobs[job_id]["status"] = "running"
    
    for i, text in enumerate(texts):
        try:
            if operation == "summarize":
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Resuma em 1 frase: {text}"}],
                    max_tokens=100
                )
                result = resp.choices[0].message.content
            
            elif operation == "classify_sentiment":
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Sentimento (positivo/negativo/neutro): {text}"}],
                    temperature=0, max_tokens=20
                )
                result = resp.choices[0].message.content.strip()
            
            else:
                result = f"Operação '{operation}' não suportada"
            
            jobs[job_id]["results"].append({"text": text[:50], "result": result})
        
        except Exception as e:
            jobs[job_id]["results"].append({"text": text[:50], "error": str(e)})
        
        jobs[job_id]["processed"] = i + 1
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["completed_at"] = datetime.now().isoformat()

# Para rodar: uvicorn pratica09.api.main:app --reload --port 8001
```

---

## 📝 Exercício 2 — Interface Streamlit

Crie `pratica09/app_streamlit.py`:

```python
# pratica09/app_streamlit.py
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import time

st.set_page_config(
    page_title="Assistente IA — IFPE",
    page_icon="🤖",
    layout="wide"
)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configurações")
    
    api_key = st.text_input("OpenAI API Key", type="password", 
                             value=st.session_state.get("api_key", ""))
    
    st.divider()
    
    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.number_input("Max tokens", 100, 2000, 500)
    
    st.divider()
    
    system_prompt = st.text_area(
        "System Prompt",
        value="Você é um assistente educacional do IFPE, especializado em IA Generativa.",
        height=100
    )
    
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────
# INTERFACE PRINCIPAL
# ──────────────────────────────────────────────

st.title("🤖 Assistente de IA Generativa")
st.caption(f"Modelo: {model} | Temperatura: {temperature}")

# Inicializar estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# Tabs
tab_chat, tab_tools, tab_stats = st.tabs(["💬 Chat", "🛠️ Ferramentas", "📊 Estatísticas"])

with tab_chat:
    # Exibir histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input
    if prompt := st.chat_input("Digite sua mensagem..."):
        if not api_key:
            st.error("Configure sua API Key na sidebar!")
        else:
            client = OpenAI(api_key=api_key)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    start = time.time()
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}] + 
                                  st.session_state.messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    latency = (time.time() - start) * 1000
                    st.caption(f"⏱️ {latency:.0f}ms")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

with tab_tools:
    st.subheader("🛠️ Ferramentas de IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Sumarizador")
        texto = st.text_area("Cole seu texto aqui:", height=150)
        max_words = st.slider("Máximo de palavras", 30, 200, 80)
        
        if st.button("Sumarizar") and texto and api_key:
            client = OpenAI(api_key=api_key)
            with st.spinner("Sumarizando..."):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Resuma em {max_words} palavras: {texto}"}]
                )
                st.success(resp.choices[0].message.content)
    
    with col2:
        st.markdown("### 🎭 Classificador de Sentimento")
        review = st.text_area("Cole o review aqui:", height=150)
        
        if st.button("Classificar") and review and api_key:
            client = OpenAI(api_key=api_key)
            with st.spinner("Classificando..."):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Classifique o sentimento (Positivo/Negativo/Neutro) e a intensidade (1-5): {review}\nRetorne JSON: {{\"sentimento\": \"\", \"intensidade\": 0, \"justificativa\": \"\"}}"}],
                    temperature=0
                )
                st.json(resp.choices[0].message.content)

with tab_stats:
    st.subheader("📊 Estatísticas da Sessão")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mensagens trocadas", len(st.session_state.messages))
    col2.metric("Modelo", model)
    col3.metric("Temperatura", temperature)
    
    if st.session_state.messages:
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
        
        avg_user = sum(len(m["content"]) for m in user_msgs) / max(1, len(user_msgs))
        avg_assistant = sum(len(m["content"]) for m in assistant_msgs) / max(1, len(assistant_msgs))
        
        st.markdown("#### Comprimento médio das mensagens")
        st.bar_chart({"Usuário": [avg_user], "Assistente": [avg_assistant]})

# Rodar: streamlit run pratica09/app_streamlit.py
```

---

## 📝 Exercício 3 — Pipeline de Análise de Texto (Projeto Principal)

Crie `pratica09/pipeline_analise.py`:

```python
# Pipeline completo: Carregar → Processar → Armazenar → Consultar → Reportar
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

load_dotenv()
client = AsyncOpenAI()

@dataclass
class DocumentoAnalisado:
    id: str
    texto: str
    sentimento: str = ""
    palavras_chave: list = field(default_factory=list)
    resumo: str = ""
    tokens_usados: int = 0
    tempo_ms: float = 0

async def analisar_documento(doc_id: str, texto: str, semaphore: asyncio.Semaphore) -> DocumentoAnalisado:
    """Analisa um documento: sentimento + keywords + resumo."""
    doc = DocumentoAnalisado(id=doc_id, texto=texto)
    start = time.time()
    
    async with semaphore:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Analise este texto e retorne JSON:
{{
  "sentimento": "positivo|negativo|neutro",
  "palavras_chave": ["p1", "p2", "p3"],
  "resumo": "máximo 20 palavras"
}}

Texto: {texto[:500]}"""
            }],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        doc.sentimento = result.get("sentimento", "")
        doc.palavras_chave = result.get("palavras_chave", [])
        doc.resumo = result.get("resumo", "")
        doc.tokens_usados = resp.usage.total_tokens
        doc.tempo_ms = (time.time() - start) * 1000
    
    return doc

async def pipeline_paralelo(textos: list[str], max_concurrent: int = 5) -> list[DocumentoAnalisado]:
    """Analisa múltiplos textos em paralelo com controle de concorrência."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tarefas = [
        analisar_documento(f"doc_{i:03d}", texto, semaphore)
        for i, texto in enumerate(textos)
    ]
    
    print(f"🚀 Processando {len(tarefas)} documentos (máx {max_concurrent} simultâneos)...")
    
    resultados = await asyncio.gather(*tarefas, return_exceptions=True)
    
    # Filtrar erros
    docs_ok = [r for r in resultados if isinstance(r, DocumentoAnalisado)]
    erros = [r for r in resultados if isinstance(r, Exception)]
    
    if erros:
        print(f"⚠️  {len(erros)} erros durante o processamento")
    
    return docs_ok

def gerar_relatorio(documentos: list[DocumentoAnalisado]) -> dict:
    """Gera relatório consolidado da análise."""
    if not documentos:
        return {}
    
    total_tokens = sum(d.tokens_usados for d in documentos)
    tempo_total = sum(d.tempo_ms for d in documentos)
    
    sentimentos = {}
    for d in documentos:
        sentimentos[d.sentimento] = sentimentos.get(d.sentimento, 0) + 1
    
    todas_keywords = []
    for d in documentos:
        todas_keywords.extend(d.palavras_chave)
    
    freq_keywords = {}
    for kw in todas_keywords:
        freq_keywords[kw] = freq_keywords.get(kw, 0) + 1
    
    top_keywords = sorted(freq_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_documentos": len(documentos),
        "total_tokens": total_tokens,
        "custo_estimado_usd": round(total_tokens * 0.00015 / 1000, 4),
        "tempo_total_ms": round(tempo_total, 2),
        "tempo_medio_ms": round(tempo_total / len(documentos), 2),
        "distribuicao_sentimento": sentimentos,
        "top_keywords": top_keywords[:5],
        "documentos": [
            {
                "id": d.id,
                "sentimento": d.sentimento,
                "resumo": d.resumo,
                "keywords": d.palavras_chave[:3]
            }
            for d in documentos
        ]
    }

async def main():
    # Textos de exemplo (em produção, carregue de arquivos/banco de dados)
    textos = [
        "Python é incrível para IA e machine learning. A comunidade é enorme e as bibliotecas são excelentes.",
        "O sistema travou três vezes hoje. Péssima experiência com esse software, definitivamente não recomendo.",
        "O produto é razoável pelo preço. Nada excepcional mas cumpre o que promete.",
        "Que aula incrível sobre RAG! Aprendi muito sobre embeddings e bancos vetoriais.",
        "Entrega atrasou 5 dias. A embalagem chegou danificada. Atendimento demorou para responder.",
        "Interface moderna e intuitiva. A curva de aprendizado é suave. Muito satisfeito.",
        "Meh. Funciona mas poderia ser melhor. Falta algumas funcionalidades básicas.",
        "Revolucionou minha produtividade! Uso todos os dias para automação de relatórios.",
    ]
    
    # Executar pipeline
    inicio = time.time()
    documentos = await pipeline_paralelo(textos, max_concurrent=4)
    tempo_total = time.time() - inicio
    
    # Gerar relatório
    relatorio = gerar_relatorio(documentos)
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO DE ANÁLISE")
    print(f"{'='*60}")
    print(f"⏱️  Tempo total (real): {tempo_total:.1f}s")
    print(f"📄 Documentos analisados: {relatorio['total_documentos']}")
    print(f"🔢 Tokens usados: {relatorio['total_tokens']}")
    print(f"💰 Custo estimado: US$ {relatorio['custo_estimado_usd']}")
    print(f"\n🎭 Sentimentos:")
    for s, c in relatorio["distribuicao_sentimento"].items():
        bar = "█" * c
        print(f"   {s:10}: {bar} ({c})")
    print(f"\n🔑 Top Keywords: {[k[0] for k in relatorio['top_keywords']]}")
    
    print(f"\n📋 Resumos:")
    for doc in relatorio["documentos"]:
        print(f"  [{doc['id']}] {doc['sentimento']:8} | {doc['resumo']}")
    
    # Salvar relatório
    with open("/tmp/relatorio_analise.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Relatório salvo em /tmp/relatorio_analise.json")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏆 Desafios Opcionais

1. **Deploy**: Dockerize a API FastAPI e suba no Railway ou Render (gratuito)
2. **Autenticação**: Adicione autenticação por API key na API FastAPI
3. **Dashboard**: No Streamlit, adicione gráficos em tempo real com `st.line_chart`
4. **Cache Redis**: Implemente cache de respostas usando `redis` ou `diskcache`

---

## ✅ Checklist de Entrega

- [ ] API FastAPI rodando com pelo menos 3 endpoints
- [ ] Interface Streamlit funcional com chat e ferramentas
- [ ] Pipeline assíncrono processando múltiplos textos
- [ ] Relatório gerado com métricas de custo e desempenho
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 08](./pratica-08-agente-ferramentas.md) | ➡️ **Próxima:** [Prática 10 — Projeto Final](./pratica-10-projeto-final.md)
