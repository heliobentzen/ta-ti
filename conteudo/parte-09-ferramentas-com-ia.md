# Parte 09 — Construindo Ferramentas com IA

> **Carga horária:** 3 horas  
> **Prática correspondente:** [Prática 09](../praticas/pratica-09-pipeline-completo.md)

---

## 9.1 Da IA para o Produto

Você já sabe os componentes individuais. Esta parte foca em **juntar tudo** para construir ferramentas reais e úteis.

> **Mindset de produto:** Uma ferramenta com IA deve resolver um problema real de forma mais eficiente do que a alternativa sem IA.

### Checklist antes de começar

- [ ] Qual problema específico isso resolve?
- [ ] Quem são os usuários?
- [ ] Qual é o input e output esperado?
- [ ] IA é a melhor solução para isso?
- [ ] Quais são os riscos de falha?
- [ ] Como medir sucesso?

---

## 9.2 Padrões de Arquitetura

### Padrão 1 — LLM como Núcleo

```
[Input] → [Preprocessamento] → [LLM] → [Postprocessamento] → [Output]
```

**Exemplos:**
- Classificador de sentimentos
- Gerador de descrições de produto
- Tradutor especializado

```python
class SentimentClassifier:
    def __init__(self):
        self.client = OpenAI()
        self.system = """Classifique o sentimento como: positivo, negativo ou neutro.
        Retorne APENAS JSON: {"sentiment": "...", "confidence": 0.0-1.0, "reason": "..."}"""
    
    def classify(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

### Padrão 2 — RAG Pipeline

```
[Documentos] → [Indexação] → [Banco Vetorial]
                                     ↑
[Query] → [Busca] → [Contexto] → [LLM] → [Resposta]
```

**Exemplos:**
- Chatbot de documentação
- Assistente de suporte técnico
- Buscador de legislação

### Padrão 3 — Agente com Ferramentas

```
[Input] → [Agente] ↔ [Ferramentas] → [Output]
```

**Exemplos:**
- Assistente que acessa banco de dados
- Agente de análise de mercado
- Automação de fluxos de trabalho

### Padrão 4 — Pipeline de Processamento

```
[Doc] → [Extração] → [Análise] → [Síntese] → [Relatório]
```

**Exemplos:**
- Análise de contratos
- Revisão de currículos
- Processamento de notas fiscais

---

## 9.3 Construindo uma API com FastAPI

```bash
pip install fastapi uvicorn python-multipart
```

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI(title="API de IA", version="1.0")
client = OpenAI()

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 500

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": request.message}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            response=result.choices[0].message.content,
            tokens_used=result.usage.total_tokens,
            model=result.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

# Iniciar: uvicorn app:app --reload
```

---

## 9.4 Interface Web com Streamlit

```bash
pip install streamlit
```

```python
# streamlit_app.py
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Chat IA", page_icon="🤖")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🤖 Assistente de IA")
st.caption("Powered by GPT-4o-mini")

# Inicializar histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua mensagem..."):
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Gerar resposta com streaming
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                stream=True
            )
            response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Iniciar: streamlit run streamlit_app.py
```

---

## 9.5 Processamento em Lote

Para processar grandes volumes de dados:

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def process_single(text: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:  # limita concorrência
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Resuma em 1 frase: {text}"}],
            max_tokens=100
        )
        return {
            "input": text[:50],
            "summary": response.choices[0].message.content
        }

async def batch_process(texts: list[str], max_concurrent: int = 10) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_single(text, semaphore) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filtrar erros
    return [r for r in results if not isinstance(r, Exception)]

# Uso
texts = ["texto 1...", "texto 2...", "texto 3..."]
results = asyncio.run(batch_process(texts))
```

---

## 9.6 Cache e Otimização

```python
import hashlib
import json
import redis  # ou shelve para local

class CachedLLM:
    def __init__(self, ttl: int = 3600):
        self.client = OpenAI()
        self.cache = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.ttl = ttl
    
    def _cache_key(self, messages: list, model: str) -> str:
        content = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def call(self, messages: list, model: str = "gpt-4o-mini", **kwargs) -> str:
        # Apenas cacheamos quando temperatura = 0 (determinístico)
        if kwargs.get("temperature", 0.7) == 0:
            cache_key = self._cache_key(messages, model)
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        response = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        result = response.choices[0].message.content
        
        if kwargs.get("temperature", 0.7) == 0:
            self.cache.setex(cache_key, self.ttl, result)
        
        return result
```

---

## 9.7 Observabilidade e Monitoramento

```python
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LLMCall:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    success: bool = True
    error: str = ""

class ObservableLLM:
    def __init__(self):
        self.client = OpenAI()
        self.calls: list[LLMCall] = []
        self.logger = logging.getLogger(__name__)
    
    def call(self, messages: list, **kwargs) -> str:
        log = LLMCall(model=kwargs.get("model", "gpt-4o-mini"))
        start = time.time()
        
        try:
            response = self.client.chat.completions.create(
                messages=messages, **kwargs
            )
            log.prompt_tokens = response.usage.prompt_tokens
            log.completion_tokens = response.usage.completion_tokens
            log.latency_ms = (time.time() - start) * 1000
            
            result = response.choices[0].message.content
            self.logger.info(f"LLM call: {log.prompt_tokens}pt + {log.completion_tokens}ct | {log.latency_ms:.0f}ms")
            return result
            
        except Exception as e:
            log.success = False
            log.error = str(e)
            log.latency_ms = (time.time() - start) * 1000
            self.logger.error(f"LLM error: {e}")
            raise
        finally:
            self.calls.append(log)
    
    def get_stats(self) -> dict:
        if not self.calls:
            return {}
        successful = [c for c in self.calls if c.success]
        return {
            "total_calls": len(self.calls),
            "success_rate": len(successful) / len(self.calls),
            "avg_latency_ms": sum(c.latency_ms for c in successful) / len(successful),
            "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in successful),
        }
```

---

## 9.8 Testes para Aplicações com IA

```python
import pytest
from unittest.mock import patch, MagicMock

# Testar com respostas mockadas (sem custo de API)
def test_sentiment_classifier():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentiment": "positivo", "confidence": 0.95}'
    
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        classifier = SentimentClassifier()
        result = classifier.classify("Adorei o produto!")
        
        assert result["sentiment"] == "positivo"
        assert result["confidence"] > 0.8

# Testes de integração com casos reais (mais lentos, custosos)
@pytest.mark.integration
def test_rag_pipeline_integration():
    rag = RAGPipeline()
    rag.index(["Python é uma linguagem de programação"])
    
    response = rag.query("O que é Python?")
    
    assert "linguagem" in response.lower()
    assert len(response) > 10
```

---

## 9.9 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Variáveis de Ambiente

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.7
    chroma_path: str = "./data/chroma"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 📌 Resumo da Parte 09

| Conceito | Descrição |
|----------|-----------|
| LLM como núcleo | Padrão simples: entrada → LLM → saída |
| FastAPI | Framework para expor modelos como APIs REST |
| Streamlit | Interface web rápida para demos e prototipagem |
| Async/Batch | Processar múltiplas requisições em paralelo |
| Cache | Evitar chamadas repetidas à API |
| Observabilidade | Monitorar latência, tokens e erros |
| Testes com mock | Testar comportamento sem chamar API real |

---

## 🔗 Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [LangSmith (observabilidade para LLMs)](https://smith.langchain.com)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

---

⬅️ **Anterior:** [Parte 08](./parte-08-agentes.md) | ➡️ **Próximo:** [Parte 10 — Projeto Final](./parte-10-projeto-final-e-tendencias.md)
