# Prática 06 — Pipeline Completo e Projeto Final

> **Carga horária estimada:** 3 horas  
> **Conteúdo relacionado:** [Parte 06](../conteudo/parte-06-ferramentas-com-ia.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Construir uma API REST com FastAPI que expõe funcionalidades de IA
- Criar uma interface web simples com Streamlit
- Construir um **Assistente Inteligente** completo integrando LLM, RAG, Agente, Memória e Observabilidade

| Componente | O que integra |
|-----------|--------------|
| 🧠 LLM | Geração de respostas inteligentes |
| 📚 RAG | Base de conhecimento do domínio |
| 🔧 Agente | Ferramentas para ações no mundo real |
| 💾 Memória | Histórico de conversa + estado |
| 🌐 Interface | Streamlit ou FastAPI |
| 📊 Observabilidade | Logs e métricas |

---

## 🔧 Setup

```bash
pip install openai fastapi uvicorn streamlit python-dotenv chromadb sentence-transformers
```

---

## 📝 Exercício 1 — API REST com FastAPI

Crie `pratica06/api/main.py`:

```python
# pratica06/api/main.py
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

# Para rodar: uvicorn pratica06.api.main:app --reload --port 8001
```

---

## 📝 Exercício 2 — Interface Streamlit

Crie `pratica06/app_streamlit.py`:

```python
# pratica06/app_streamlit.py
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

# Rodar: streamlit run pratica06/app_streamlit.py
```

---

## 📝 Exercício 3 — Projeto Final: Assistente Inteligente

Neste exercício você construirá um assistente completo integrando todos os conceitos do curso.

### 🏗️ Estrutura do Projeto

```
pratica06/
├── assistente/
│   ├── __init__.py
│   ├── config.py          ← Configurações centralizadas
│   ├── knowledge_base.py  ← Gerenciamento da base RAG
│   ├── tools.py           ← Ferramentas do agente
│   ├── agent.py           ← Agente principal
│   └── observability.py   ← Logs e métricas
├── interface/
│   ├── cli.py             ← Interface CLI
│   └── app.py             ← Interface Streamlit
├── api/
│   └── main.py            ← API FastAPI (exercício 1)
├── app_streamlit.py       ← Interface Streamlit (exercício 2)
├── dados/
│   └── conhecimento/      ← Arquivos de conhecimento
└── README.md              ← Documentação do seu projeto
```

### Passo 3.1 — Configuração Central

Crie `pratica06/assistente/config.py`:

```python
# pratica06/assistente/config.py
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1000
    
    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "./pratica06/dados/chroma"
    collection_name: str = "assistente_kb"
    rag_top_k: int = 3
    
    # Agente
    max_steps: int = 8
    
    # Interface
    assistant_name: str = "Assistente IFPE TA-TI"
    assistant_persona: str = """Você é um assistente educacional especializado em IA Generativa.
Ajude os alunos a entender e aplicar os conceitos do curso.
Seja didático, use exemplos e incentive a prática.
Quando não souber algo, admita e sugira onde buscar."""
    
    # Observabilidade
    log_file: str = "./pratica06/dados/assistente.log"

config = Config()
```

### Passo 3.2 — Base de Conhecimento

Crie `pratica06/assistente/knowledge_base.py`:

```python
# pratica06/assistente/knowledge_base.py
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import os
from .config import config

class KnowledgeBase:
    """Gerenciador da base de conhecimento RAG."""
    
    def __init__(self):
        os.makedirs(config.chroma_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=config.chroma_path)
        
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        
        self.collection = self._client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, text: str, source: str, metadata: dict = None) -> str:
        """Adiciona um documento (com chunking automático)."""
        chunks = self._chunk(text)
        
        added = 0
        for i, chunk in enumerate(chunks):
            doc_id = f"{source}_chunk{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            
            meta = {"source": source, "chunk": i, "total_chunks": len(chunks)}
            if metadata:
                meta.update(metadata)
            
            self.collection.upsert(
                documents=[chunk],
                ids=[doc_id],
                metadatas=[meta]
            )
            added += 1
        
        return f"Adicionado: {source} ({added} chunks)"
    
    def add_documents_batch(self, documents: list[dict]) -> int:
        """
        Adiciona múltiplos documentos.
        documents: lista de {"text": str, "source": str, "metadata": dict}
        """
        total = 0
        for doc in documents:
            self.add_document(doc["text"], doc["source"], doc.get("metadata"))
            total += 1
        return total
    
    def search(self, query: str, n: int = None, filter_meta: dict = None) -> list[dict]:
        """Busca documentos relevantes."""
        if self.collection.count() == 0:
            return []
        
        n = n or config.rag_top_k
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, self.collection.count()),
            where=filter_meta
        )
        
        return [
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "score": round(1 - dist, 3)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def get_context(self, query: str) -> str:
        """Retorna contexto formatado para o prompt."""
        docs = self.search(query)
        if not docs:
            return ""
        
        return "\n\n".join([
            f"[{d['source']} | relevância: {d['score']}]\n{d['text']}"
            for d in docs
        ])
    
    def stats(self) -> dict:
        return {"total_chunks": self.collection.count()}
    
    @staticmethod
    def _chunk(text: str, size: int = 400, overlap: int = 50) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                chunks.append(chunk)
            start = end - overlap
        return chunks
```

### Passo 3.3 — Ferramentas do Agente

Crie `pratica06/assistente/tools.py`:

```python
# pratica06/assistente/tools.py
import json
from datetime import datetime

# ──────────────────────────────────────────────
# FERRAMENTAS BASE (funcionam em qualquer domínio)
# ──────────────────────────────────────────────

def get_current_datetime() -> dict:
    """Retorna data e hora atual."""
    now = datetime.now()
    return {
        "data": now.strftime("%d/%m/%Y"),
        "hora": now.strftime("%H:%M"),
        "dia_semana": ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"][now.weekday()]
    }

def calcular(expressao: str) -> str:
    """Avalia expressão matemática simples."""
    try:
        resultado = eval(expressao, {"__builtins__": {}})
        return f"{expressao} = {resultado}"
    except Exception as e:
        return f"Erro ao calcular '{expressao}': {e}"

def formatar_codigo(codigo: str, linguagem: str = "python") -> str:
    """Formata e valida um bloco de código."""
    return f"```{linguagem}\n{codigo.strip()}\n```"

# ──────────────────────────────────────────────
# FERRAMENTAS ESPECÍFICAS DO DOMÍNIO
# (adicione as suas aqui)
# ──────────────────────────────────────────────

CONTEUDO_CURSO = {
    "parte01": {"titulo": "Introdução à IA Generativa", "carga": "2h"},
    "parte02": {"titulo": "LLMs: Como Funcionam", "carga": "3h"},
    "parte03": {"titulo": "APIs de LLMs", "carga": "2h"},
    "parte04": {"titulo": "Prompt Engineering", "carga": "2h"},
    "parte05": {"titulo": "Embeddings", "carga": "3h"},
    "parte06": {"titulo": "Bancos Vetoriais", "carga": "2h"},
    "parte07": {"titulo": "RAG", "carga": "3h"},
    "parte08": {"titulo": "Agentes de IA", "carga": "3h"},
    "parte09": {"titulo": "Ferramentas com IA", "carga": "3h"},
    "parte10": {"titulo": "Projeto Final", "carga": "3h"},
}

def listar_partes_curso() -> list:
    """Lista todas as partes do curso."""
    return [{"id": k, **v} for k, v in CONTEUDO_CURSO.items()]

def buscar_parte_curso(termo: str) -> list:
    """Busca partes do curso por termo."""
    resultados = []
    for id_, info in CONTEUDO_CURSO.items():
        if termo.lower() in info["titulo"].lower():
            resultados.append({"id": id_, **info})
    return resultados if resultados else [{"info": f"Nenhuma parte encontrada para '{termo}'"}]

# ──────────────────────────────────────────────
# SCHEMA OPENAI E MAPA DE FUNÇÕES
# ──────────────────────────────────────────────

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_current_datetime",
        "description": "Retorna a data e hora atual",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "calcular",
        "description": "Calcula uma expressão matemática",
        "parameters": {"type": "object", "properties": {
            "expressao": {"type": "string", "description": "Ex: 2 * 50 + 10"}
        }, "required": ["expressao"]}
    }},
    {"type": "function", "function": {
        "name": "listar_partes_curso",
        "description": "Lista todas as partes do curso de IA Generativa",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "buscar_parte_curso",
        "description": "Busca partes do curso por tema ou palavra-chave",
        "parameters": {"type": "object", "properties": {
            "termo": {"type": "string"}
        }, "required": ["termo"]}
    }},
]

TOOLS_MAP = {
    "get_current_datetime": get_current_datetime,
    "calcular": calcular,
    "listar_partes_curso": listar_partes_curso,
    "buscar_parte_curso": buscar_parte_curso,
}
```

### Passo 3.4 — Observabilidade

Crie `pratica06/assistente/observability.py`:

```python
# pratica06/assistente/observability.py
import json
import logging
from datetime import datetime
from .config import config
import os

os.makedirs(os.path.dirname(config.log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(config.log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

class Logger:
    def __init__(self):
        self.logger = logging.getLogger("assistente")
        self._total_tokens = 0
        self._total_requests = 0
        self._tool_calls = []
    
    def log_request(self, message: str):
        self._total_requests += 1
        self.logger.info(f"REQUEST #{self._total_requests}: {message[:100]}")
    
    def log_response(self, response: str):
        self.logger.info(f"RESPONSE ({len(response)} chars)")
    
    def log_tokens(self, tokens: int):
        self._total_tokens += tokens
    
    def log_tool_call(self, name: str, args: dict, result):
        self._tool_calls.append({"tool": name, "args": args, "ts": datetime.now().isoformat()})
        self.logger.info(f"TOOL: {name}({json.dumps(args)[:50]}) → {str(result)[:50]}")
    
    def get_stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "custo_estimado_usd": round(self._total_tokens * 0.00015 / 1000, 4),
            "total_tool_calls": len(self._tool_calls)
        }
```

### Passo 3.5 — Agente Principal

Crie `pratica06/assistente/agent.py`:

```python
# pratica06/assistente/agent.py
from openai import OpenAI
import json
from .config import config
from .knowledge_base import KnowledgeBase
from .tools import TOOLS_SCHEMA, TOOLS_MAP
from .observability import Logger

class Assistente:
    """Assistente inteligente com RAG + Agente + Memória."""
    
    def __init__(self):
        self.client = OpenAI()
        self.kb = KnowledgeBase()
        self.logger = Logger()
        self.history = []
        self._step_count = 0
    
    def carregar_conhecimento(self, documentos: list[dict]) -> str:
        """Carrega documentos na base de conhecimento."""
        total = self.kb.add_documents_batch(documentos)
        return f"✅ {total} documentos carregados ({self.kb.stats()['total_chunks']} chunks)"
    
    def chat(self, mensagem: str) -> str:
        """Processa uma mensagem e retorna resposta."""
        self.logger.log_request(mensagem)
        
        # Buscar contexto RAG
        contexto = self.kb.get_context(mensagem)
        
        # System prompt com contexto
        system = config.assistant_persona
        if contexto:
            system += f"\n\n📚 CONHECIMENTO RELEVANTE:\n{contexto}"
        
        # Montar mensagens
        messages = [{"role": "system", "content": system}]
        messages += self.history[-8:]  # últimas 4 trocas
        messages += [{"role": "user", "content": mensagem}]
        
        # Loop do agente
        resposta = self._run_agent_loop(messages)
        
        # Atualizar histórico
        self.history.append({"role": "user", "content": mensagem})
        self.history.append({"role": "assistant", "content": resposta})
        
        self.logger.log_response(resposta)
        return resposta
    
    def _run_agent_loop(self, messages: list) -> str:
        """Loop principal do agente com ferramentas."""
        for step in range(config.max_steps):
            resp = self.client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            
            msg = resp.choices[0].message
            self._step_count += 1
            self.logger.log_tokens(resp.usage.total_tokens)
            
            if resp.choices[0].finish_reason == "stop":
                return msg.content
            
            if resp.choices[0].finish_reason == "tool_calls":
                messages.append(msg)
                for tc in msg.tool_calls:
                    nome = tc.function.name
                    args = json.loads(tc.function.arguments)
                    
                    if nome in TOOLS_MAP:
                        resultado = TOOLS_MAP[nome](**args)
                    else:
                        resultado = {"erro": f"Ferramenta '{nome}' não encontrada"}
                    
                    self.logger.log_tool_call(nome, args, resultado)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(resultado, ensure_ascii=False)
                    })
        
        return "Desculpe, não consegui processar sua solicitação. Pode reformular?"
    
    def limpar_historico(self):
        self.history = []
    
    def estatisticas(self) -> dict:
        return {
            **self.logger.get_stats(),
            **self.kb.stats(),
            "history_length": len(self.history)
        }
```

### Passo 3.6 — Interface CLI

Crie `pratica06/interface/cli.py`:

```python
# pratica06/interface/cli.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pratica06.assistente.agent import Assistente

def main():
    assistente = Assistente()
    
    # Carregar base de conhecimento inicial
    docs_iniciais = [
        {
            "text": "RAG (Retrieval Augmented Generation) é uma arquitetura que combina busca de documentos com geração de texto por LLMs.",
            "source": "curso_ia",
            "metadata": {"parte": "07", "topico": "rag"}
        },
        {
            "text": "Embeddings são representações vetoriais de texto. Textos semanticamente similares têm vetores próximos no espaço vetorial.",
            "source": "curso_ia",
            "metadata": {"parte": "05", "topico": "embeddings"}
        },
        {
            "text": "Agentes de IA usam LLMs para raciocinar e executar ações usando ferramentas (function calling).",
            "source": "curso_ia",
            "metadata": {"parte": "08", "topico": "agentes"}
        },
    ]
    
    msg = assistente.carregar_conhecimento(docs_iniciais)
    
    print(f"\n🤖 {assistente.kb.stats()['total_chunks']} chunks na base de conhecimento")
    print(f"\n{'='*55}")
    print(f"  Bem-vindo ao Assistente IA — IFPE TA-TI")
    print(f"  Comandos: /sair | /limpar | /stats | /adicionar")
    print(f"{'='*55}\n")
    
    while True:
        try:
            entrada = input("Você: ").strip()
            
            if not entrada:
                continue
            
            if entrada == "/sair":
                stats = assistente.estatisticas()
                print(f"\n📊 Sessão encerrada:")
                print(f"   Requisições: {stats['total_requests']}")
                print(f"   Tokens: {stats['total_tokens']}")
                print(f"   Custo estimado: US$ {stats['custo_estimado_usd']}")
                break
            
            elif entrada == "/limpar":
                assistente.limpar_historico()
                print("🗑️  Histórico limpo!\n")
            
            elif entrada == "/stats":
                stats = assistente.estatisticas()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                print()
            
            elif entrada.startswith("/adicionar "):
                texto = entrada[11:].strip()
                if texto:
                    msg = assistente.carregar_conhecimento([
                        {"text": texto, "source": "usuario", "metadata": {"tipo": "manual"}}
                    ])
                    print(f"✅ {msg}\n")
            
            else:
                print(f"\n🤖 ", end="", flush=True)
                resposta = assistente.chat(entrada)
                print(f"{resposta}\n")
        
        except KeyboardInterrupt:
            print("\n\nEncerrando...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}\n")

if __name__ == "__main__":
    main()
```

### Passo 3.7 — Documentação do Seu Projeto

Crie `pratica06/README.md` com:

```markdown
# [Nome do Seu Assistente]

> Projeto Final — Disciplina Tópicos Avançados em TI  
> IFPE | Professor Hélio Bentzen  
> Aluno: [Seu nome]

## 🎯 Domínio Escolhido

[Descreva o domínio: o que o assistente faz, para quem é destinado]

## 🏗️ Arquitetura

[Diagrama ou descrição dos componentes]

## 🚀 Como Executar

\`\`\`bash
pip install -r requirements.txt
python -m pratica06.interface.cli
\`\`\`

## 📚 Base de Conhecimento

[Liste os documentos/fontes que foram indexados]

## 🛠️ Ferramentas Implementadas

| Ferramenta | Descrição |
|-----------|-----------|
| ... | ... |

## 📊 Resultados e Análise

[O que funcionou bem? O que poderia melhorar?]

## 🔮 Próximos Passos

[O que você implementaria se tivesse mais tempo?]
```

---

## 🏆 Critérios de Avaliação

| Critério | Peso | Descrição |
|---------|------|-----------|
| RAG implementado | 20% | Base de conhecimento indexada e buscada |
| Agente com ferramentas | 20% | Pelo menos 3 ferramentas funcionando |
| Qualidade das respostas | 20% | Respostas coerentes e baseadas no contexto |
| Código organizado | 15% | Estrutura de arquivos, funções bem definidas |
| Observabilidade | 10% | Logs e métricas implementados |
| Interface funcional | 10% | CLI ou web usável |
| Documentação | 5% | README completo |

---

## ✅ Checklist de Entrega

- [ ] API FastAPI rodando com pelo menos 3 endpoints
- [ ] Interface Streamlit funcional com chat e ferramentas
- [ ] Estrutura de pastas do assistente criada conforme o guia
- [ ] Base de conhecimento com pelo menos 10 documentos relevantes ao domínio
- [ ] Pelo menos 3 ferramentas específicas do domínio implementadas
- [ ] Interface CLI ou web funcional
- [ ] Logs sendo gerados
- [ ] README documentando o projeto
- [ ] Demonstração ao vivo funcionando

---

## 🎉 Parabéns!

Se você chegou até aqui, completou a jornada de 26 horas sobre IA Generativa para Programadores. Você agora tem as ferramentas para construir aplicações poderosas com IA!

> *"A melhor forma de entender IA é construindo com ela."* — Prof. Hélio Bentzen

---

⬅️ **Anterior:** [Prática 05](./pratica-05-rag-simples.md)  
🏠 **Início:** [README Principal](../README.md)
