# Prática 06 — RAG Avançado com Reranking e Hybrid Search

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 07](../conteudo/parte-07-rag.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Implementar query rewriting para melhorar a recuperação
- Usar reranking para reordenar resultados com maior precisão
- Combinar busca vetorial e por palavras-chave (hybrid search)
- Avaliar a qualidade do RAG com métricas objetivas

---

## 🔧 Setup

```bash
pip install openai chromadb python-dotenv sentence-transformers rank-bm25 numpy
```

---

## 📝 Exercício 1 — Query Rewriting

Crie `pratica06/ex01_query_rewriting.py`:

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Base de conhecimento técnica
DOCS = [
    "Para autenticar na API REST, envie o Bearer token no header Authorization: Bearer {token}",
    "Rate limiting: máximo 100 requisições por minuto por API key. Respostas 429 indicam limite excedido.",
    "Webhooks são callbacks HTTP que a API envia quando eventos ocorrem no sistema.",
    "Paginação usa cursor-based: parâmetros 'cursor' e 'limit'. Retorna 'next_cursor' se há mais páginas.",
    "Erros retornam JSON com campos 'error_code', 'message' e 'details'.",
    "Timeout padrão: 30 segundos. Para operações longas, use o endpoint assíncrono com job_id.",
]

chroma = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = chroma.create_collection("docs_api", embedding_function=ef)
col.add(documents=DOCS, ids=[f"d{i}" for i in range(len(DOCS))])

def rewrite_query(query: str) -> str:
    """Reescreve a query para melhorar a recuperação."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Reescreva esta pergunta para busca em documentação técnica.
            Expanda siglas, adicione termos técnicos relacionados, seja mais específico.
            Retorne APENAS a pergunta reescrita, sem explicações.
            
            Pergunta original: {query}"""
        }],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def buscar(query: str, rewrite: bool = False, n: int = 3):
    """Busca documentos com ou sem rewriting."""
    query_usada = rewrite_query(query) if rewrite else query
    results = col.query(query_texts=[query_usada], n_results=n)
    return query_usada, results["documents"][0], results["distances"][0]

# Perguntas com linguagem informal
perguntas = [
    "como me autenticar?",
    "minha request tá dando 429",
    "como saber se tem mais dados?",
    "quanto tempo esperar antes de dar timeout?",
]

for p in perguntas:
    print(f"\n{'='*60}")
    print(f"❓ Original: {p}")
    
    q_orig, docs_orig, dists_orig = buscar(p, rewrite=False)
    q_rw, docs_rw, dists_rw = buscar(p, rewrite=True)
    
    print(f"📝 Reescrita: {q_rw}")
    print(f"\nTop resultado SEM rewrite  [{1-dists_orig[0]:.3f}]: {docs_orig[0][:70]}...")
    print(f"Top resultado COM rewrite  [{1-dists_rw[0]:.3f}]: {docs_rw[0][:70]}...")
```

---

## 📝 Exercício 2 — Reranking com Cross-Encoder

Crie `pratica06/ex02_reranking.py`:

```python
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils import embedding_functions

# Cross-encoder para reranking (mais preciso que bi-encoder)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

DOCS = [
    "Python é uma linguagem de programação de alto nível criada por Guido van Rossum em 1991.",
    "Python suporta múltiplos paradigmas: orientado a objetos, funcional e imperativo.",
    "A linguagem Python é amplamente usada em ciência de dados, machine learning e web.",
    "PEP 8 é o guia de estilo oficial para código Python, definindo convenções de formatação.",
    "Python tem gerenciamento automático de memória através do garbage collector.",
    "A filosofia Python está no Zen of Python: 'Legibilidade conta', 'Simples é melhor que complexo'.",
    "Pip é o gerenciador de pacotes oficial do Python, usado para instalar bibliotecas.",
    "Ambientes virtuais (venv) isolam dependências entre projetos Python diferentes.",
    "Python é interpretado, o que facilita debugging mas pode ser mais lento que linguagens compiladas.",
    "Type hints foram introduzidos no Python 3.5 para adicionar anotações de tipo estáticas.",
]

# Setup
chroma = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = chroma.create_collection("python_docs", embedding_function=ef)
col.add(documents=DOCS, ids=[f"d{i}" for i in range(len(DOCS))])

def buscar_sem_rerank(query: str, top_k: int = 5):
    """Busca vetorial simples."""
    results = col.query(query_texts=[query], n_results=top_k)
    return list(zip(results["documents"][0], results["distances"][0]))

def buscar_com_rerank(query: str, candidates: int = 8, top_k: int = 3):
    """Busca vetorial + reranking."""
    # 1. Recuperar mais candidatos do que o necessário
    results = col.query(query_texts=[query], n_results=candidates)
    candidate_docs = results["documents"][0]
    
    # 2. Rerankar com cross-encoder (mais preciso)
    pairs = [[query, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)
    
    # 3. Ordenar por score do reranker
    ranked = sorted(zip(scores, candidate_docs), reverse=True)
    return ranked[:top_k]

# Comparar
queries = [
    "Como organizar projetos Python profissionalmente?",
    "Python é rápido?",
    "Filosofia e boas práticas de Python",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"❓ {q}\n")
    
    sem_rr = buscar_sem_rerank(q, top_k=3)
    com_rr = buscar_com_rerank(q, candidates=8, top_k=3)
    
    print("SEM Reranking:")
    for doc, dist in sem_rr:
        print(f"  [{1-dist:.3f}] {doc[:70]}...")
    
    print("\nCOM Reranking:")
    for score, doc in com_rr:
        print(f"  [{score:.3f}] {doc[:70]}...")
```

---

## 📝 Exercício 3 — Hybrid Search (BM25 + Vetorial)

Crie `pratica06/ex03_hybrid_search.py`:

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

DOCUMENTOS = [
    "Python para machine learning: bibliotecas NumPy, Pandas, Scikit-learn e TensorFlow.",
    "JavaScript e Node.js para desenvolvimento web e APIs REST.",
    "Docker e Kubernetes para containerização e orquestração de microsserviços.",
    "SQL e PostgreSQL para bancos de dados relacionais e consultas complexas.",
    "Git e GitHub para controle de versão e colaboração em equipe.",
    "RAG com LLMs: combina busca vetorial com geração de texto por modelos de linguagem.",
    "Fine-tuning de LLMs com LoRA e QLoRA para adaptação a domínios específicos.",
    "APIs REST com FastAPI: alta performance, type hints, documentação automática.",
    "Agentes de IA com LangChain: orquestração de ferramentas e cadeia de raciocínio.",
    "Embeddings e bancos vetoriais: ChromaDB, Pinecone, pgvector para busca semântica.",
]

# Modelo para embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Preparar BM25 (busca léxica)
def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

tokenized_docs = [tokenize(doc) for doc in DOCUMENTOS]
bm25 = BM25Okapi(tokenized_docs)

# Preparar embeddings (busca semântica)
doc_embeddings = model.encode(DOCUMENTOS)

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normaliza scores para [0, 1]."""
    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)

def hybrid_search(query: str, top_k: int = 3, alpha: float = 0.5) -> list:
    """
    Hybrid search: combina BM25 e vetorial.
    alpha=0 → apenas BM25 | alpha=1 → apenas vetorial | alpha=0.5 → igualmente
    """
    # BM25 scores
    bm25_scores = np.array(bm25.get_scores(tokenize(query)))
    
    # Vetorial scores
    query_emb = model.encode([query])
    vec_scores = cosine_similarity(query_emb, doc_embeddings)[0]
    
    # Normalizar e combinar
    bm25_norm = normalize_scores(bm25_scores)
    vec_norm = normalize_scores(vec_scores)
    
    combined = (1 - alpha) * bm25_norm + alpha * vec_norm
    
    # Top-k
    top_indices = combined.argsort()[-top_k:][::-1]
    
    return [
        {
            "doc": DOCUMENTOS[i],
            "score_combinado": combined[i],
            "score_bm25": bm25_norm[i],
            "score_vetorial": vec_norm[i]
        }
        for i in top_indices
    ]

# Testar com diferentes valores de alpha
queries_teste = [
    "LLM e RAG para busca inteligente",    # semântico
    "Python NumPy Pandas",                  # léxico (palavras exatas)
    "containerização kubernetes docker",    # híbrido
]

for query in queries_teste:
    print(f"\n{'='*60}")
    print(f"❓ {query}\n")
    
    for alpha in [0.0, 0.5, 1.0]:
        nome = "BM25 puro" if alpha == 0 else "Vetorial puro" if alpha == 1 else "Híbrido 50/50"
        resultado = hybrid_search(query, top_k=1, alpha=alpha)
        r = resultado[0]
        print(f"  {nome} [{r['score_combinado']:.3f}]: {r['doc'][:60]}...")
```

---

## 📝 Exercício 4 — Pipeline RAG Avançado Completo (Projeto Principal)

Crie `pratica06/rag_avancado.py`:

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()
llm_client = OpenAI()

class RAGAvancado:
    """
    Pipeline RAG avançado com:
    - Query rewriting
    - Hybrid search (BM25 + vetorial)
    - Reranking com cross-encoder
    """
    
    def __init__(self, documentos: list[str]):
        self.documentos = documentos
        self.bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Indexar para BM25
        self._tokenize = lambda t: re.findall(r'\w+', t.lower())
        self.bm25 = BM25Okapi([self._tokenize(d) for d in documentos])
        
        # Indexar para vetorial
        print("📊 Gerando embeddings dos documentos...")
        self.doc_embeddings = self.bi_encoder.encode(documentos)
        print(f"✅ {len(documentos)} documentos indexados")
    
    def _rewrite_query(self, query: str) -> str:
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Reescreva para busca técnica, sem explicações: {query}"
            }],
            temperature=0,
            max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    
    def _hybrid_retrieve(self, query: str, candidates: int = 10) -> list[str]:
        # BM25
        bm25_scores = np.array(self.bm25.get_scores(self._tokenize(query)))
        
        # Vetorial
        q_emb = self.bi_encoder.encode([query])
        vec_scores = cosine_similarity(q_emb, self.doc_embeddings)[0]
        
        # Combinar
        def norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-9)
        
        combined = 0.4 * norm(bm25_scores) + 0.6 * norm(vec_scores)
        top_idx = combined.argsort()[-candidates:][::-1]
        
        return [self.documentos[i] for i in top_idx]
    
    def _rerank(self, query: str, docs: list[str], top_k: int = 3) -> list[str]:
        pairs = [[query, doc] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, docs), reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    
    def query(self, pergunta: str, verbose: bool = False) -> str:
        """Pipeline completo: rewrite → retrieve → rerank → generate."""
        
        # 1. Rewrite
        query_reescrita = self._rewrite_query(pergunta)
        if verbose:
            print(f"  📝 Query reescrita: {query_reescrita}")
        
        # 2. Hybrid retrieve (candidatos)
        candidatos = self._hybrid_retrieve(query_reescrita, candidates=8)
        if verbose:
            print(f"  🔍 {len(candidatos)} candidatos recuperados")
        
        # 3. Rerank
        top_docs = self._rerank(query_reescrita, candidatos, top_k=3)
        if verbose:
            print(f"  ✅ Top 3 após reranking selecionados")
        
        # 4. Generate
        contexto = "\n\n".join(top_docs)
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Use o contexto abaixo para responder. Se não souber, diga.

CONTEXTO:
{contexto}"""
                },
                {"role": "user", "content": pergunta}
            ],
            temperature=0.2
        )
        
        return resp.choices[0].message.content

# ──────────── DEMO ────────────

TECH_DOCS = [
    "Python é usado em IA, data science, web com Django/Flask, automação e scripts.",
    "FastAPI é um framework Python moderno para APIs REST com alta performance e suporte a async.",
    "Docker containeriza aplicações garantindo ambiente consistente em qualquer máquina.",
    "PostgreSQL é um banco relacional robusto com suporte a JSON, arrays e full-text search.",
    "RAG (Retrieval Augmented Generation) conecta LLMs a bases de conhecimento para respostas precisas.",
    "Kubernetes orquestra containers Docker em escala, com auto-scaling e self-healing.",
    "Git é o sistema de controle de versão mais usado, com GitHub como plataforma principal.",
    "LangChain facilita a criação de aplicações com LLMs, chains, agents e memory.",
    "Embeddings são representações vetoriais de textos que capturam significado semântico.",
    "Agentes de IA podem usar ferramentas como busca web, calculadora e APIs externas.",
]

rag = RAGAvancado(TECH_DOCS)

perguntas = [
    "Como construir uma API rápida em Python?",
    "Qual tecnologia usar pra dar escala pra minha aplicação?",
    "Como criar um chatbot que consulta meus documentos?",
]

print("\n" + "="*60)
for p in perguntas:
    print(f"\n❓ {p}")
    resposta = rag.query(p, verbose=True)
    print(f"\n💬 {resposta}")
    print("-"*60)
```

---

## 🏆 Desafios Opcionais

1. **Multi-Query RAG**: Gere 3 variações da pergunta, recupere para cada uma e combine os resultados únicos
2. **RAG com citações**: Peça ao modelo para citar qual trecho específico embasou cada afirmação
3. **Avaliação automática**: Implemente RAGAS para medir Faithfulness e Answer Relevancy
4. **Streaming**: Adicione streaming de tokens na geração da resposta final

---

## ✅ Checklist de Entrega

- [ ] Ex01: query rewriting funcionando
- [ ] Ex02: reranking com cross-encoder comparado
- [ ] Ex03: hybrid search com alpha variável
- [ ] Pipeline RAG avançado completo funcionando
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 05](./pratica-05-rag-simples.md) | ➡️ **Próxima:** [Prática 07 — Agente Simples](./pratica-07-agente-simples.md)
