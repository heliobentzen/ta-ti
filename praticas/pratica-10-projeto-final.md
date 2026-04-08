# Prática 10 — Projeto Final: Assistente Inteligente Completo

> **Carga horária estimada:** 3 horas  
> **Conteúdo relacionado:** [Parte 10](../conteudo/parte-10-projeto-final-e-tendencias.md)

---

## 🎯 Objetivo do Projeto Final

Construir um **Assistente Inteligente** completo que integra **todos os conceitos** aprendidos na disciplina:

| Componente | O que integra |
|-----------|--------------|
| 🧠 LLM | Geração de respostas inteligentes |
| 📚 RAG | Base de conhecimento do domínio |
| 🔧 Agente | Ferramentas para ações no mundo real |
| 💾 Memória | Histórico de conversa + estado |
| 🌐 Interface | Streamlit ou FastAPI |
| 📊 Observabilidade | Logs e métricas |

---

## 🏗️ Estrutura do Projeto

```
pratica10/
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
├── dados/
│   └── conhecimento/      ← Arquivos de conhecimento
└── README.md              ← Documentação do seu projeto
```

---

## 📝 Passo 1 — Configuração Central

Crie `pratica10/assistente/config.py`:

```python
# pratica10/assistente/config.py
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
    chroma_path: str = "./pratica10/dados/chroma"
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
    log_file: str = "./pratica10/dados/assistente.log"

config = Config()
```

---

## 📝 Passo 2 — Base de Conhecimento

Crie `pratica10/assistente/knowledge_base.py`:

```python
# pratica10/assistente/knowledge_base.py
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

---

## 📝 Passo 3 — Ferramentas do Agente

Crie `pratica10/assistente/tools.py`:

```python
# pratica10/assistente/tools.py
# Defina aqui as ferramentas específicas do seu domínio
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

# Exemplo para domínio educacional:
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

---

## 📝 Passo 4 — Agente Principal

Crie `pratica10/assistente/agent.py`:

```python
# pratica10/assistente/agent.py
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

---

## 📝 Passo 5 — Observabilidade

Crie `pratica10/assistente/observability.py`:

```python
# pratica10/assistente/observability.py
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

---

## 📝 Passo 6 — Interface CLI

Crie `pratica10/interface/cli.py`:

```python
# pratica10/interface/cli.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pratica10.assistente.agent import Assistente

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

---

## 📝 Passo 7 — Documentação do Seu Projeto

Crie `pratica10/README.md` com:

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
python -m pratica10.interface.cli
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

- [ ] Estrutura de pastas criada conforme o guia
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

⬅️ **Anterior:** [Prática 09](./pratica-09-pipeline-completo.md)  
🏠 **Início:** [README Principal](../README.md)
