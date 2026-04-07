# Parte 07 — RAG: Retrieval Augmented Generation

> **Carga horária:** 3 horas  
> **Práticas correspondentes:** [Prática 05](../praticas/pratica-05-rag-simples.md) e [Prática 06](../praticas/pratica-06-rag-avancado.md)

---

## 7.1 O que é RAG?

**RAG (Retrieval Augmented Generation)** é uma arquitetura que combina:
1. **Recuperação** de informações relevantes de uma base de conhecimento
2. **Geração** de resposta usando um LLM com esse contexto recuperado

> **Problema que resolve:** LLMs têm conhecimento estático (data de corte de treinamento) e não conhecem seus dados específicos (documentos internos, bases de dados privadas).

```
Sem RAG:
Usuário → "Qual é a política de férias da empresa?" → LLM → "Não tenho essa informação"

Com RAG:
Usuário → "Qual é a política de férias?" 
    → Busca em documentos da empresa
    → Encontra: "Art. 3: Colaboradores têm 30 dias de férias por ano..."
    → LLM recebe o contexto + pergunta
    → "Segundo a política da empresa, você tem direito a 30 dias de férias..."
```

---

## 7.2 Arquitetura RAG

```
[Documentos] → [Chunking] → [Embedding] → [Banco Vetorial]
                                                    ↑
[Usuário] → [Query] → [Embedding da Query] → [Busca ANN]
                                                    ↓
                                         [Chunks Relevantes]
                                                    ↓
                                    [Prompt = Query + Contexto]
                                                    ↓
                                              [LLM]
                                                    ↓
                                             [Resposta]
```

### Duas Fases

**Fase 1 — Indexação** (feita uma vez ou periodicamente):
1. Carregamento dos documentos
2. Chunking (divisão em pedaços)
3. Geração de embeddings
4. Armazenamento no banco vetorial

**Fase 2 — Consulta** (feita a cada pergunta):
1. Geração do embedding da pergunta
2. Busca dos chunks mais relevantes
3. Construção do prompt com contexto
4. Geração da resposta pelo LLM

---

## 7.3 RAG Simples — Implementação do Zero

```python
from openai import OpenAI
import chromadb
import os

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("knowledge_base", 
                                              metadata={"hnsw:space": "cosine"})

# ============================================================
# FASE 1: INDEXAÇÃO
# ============================================================

def load_and_index(documents: list[dict]):
    """
    documents: lista de {"id": str, "content": str, "source": str}
    """
    texts = [d["content"] for d in documents]
    ids = [d["id"] for d in documents]
    metadatas = [{"source": d["source"]} for d in documents]
    
    # Gerar embeddings em lote
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]
    
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ {len(documents)} documentos indexados")

# Documentos de exemplo
docs = [
    {"id": "politica_ferias", "source": "rh.pdf",
     "content": "Política de Férias: Todos os colaboradores têm direito a 30 dias de férias anuais após 12 meses de trabalho."},
    {"id": "politica_home", "source": "rh.pdf", 
     "content": "Home Office: Colaboradores podem trabalhar remotamente até 3 dias por semana mediante aprovação do gestor."},
    {"id": "beneficios", "source": "rh.pdf",
     "content": "Benefícios: Vale alimentação R$600/mês, plano de saúde Unimed, gympass categoria prata."},
    {"id": "horario", "source": "rh.pdf",
     "content": "Horário de trabalho: 8h às 17h com 1h de almoço. Banco de horas disponível."},
]

load_and_index(docs)

# ============================================================
# FASE 2: CONSULTA
# ============================================================

def rag_query(user_question: str, top_k: int = 3) -> str:
    """Pipeline RAG completo."""
    
    # 1. Embedding da pergunta
    query_response = client.embeddings.create(
        input=user_question,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding
    
    # 2. Buscar chunks relevantes
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0]
    sources = [m.get("source", "?") for m in results["metadatas"][0]]
    
    # 3. Construir contexto
    context = "\n\n".join([
        f"[Fonte: {src}]\n{doc}" 
        for doc, src in zip(retrieved_docs, sources)
    ])
    
    # 4. Prompt com contexto
    prompt = f"""Use APENAS as informações do contexto abaixo para responder.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."

CONTEXTO:
{context}

PERGUNTA: {user_question}

RESPOSTA:"""
    
    # 5. Gerar resposta
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente de RH preciso e direto."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

# Testar
print(rag_query("Quantos dias de férias tenho?"))
print(rag_query("Posso trabalhar de casa?"))
print(rag_query("Qual é o salário?"))  # não está na base
```

---

## 7.4 Avaliação de RAG

### Métricas Principais

| Métrica | Descrição |
|---------|-----------|
| **Faithfulness** | A resposta é fiel ao contexto recuperado? |
| **Answer Relevancy** | A resposta é relevante para a pergunta? |
| **Context Recall** | Os documentos relevantes foram recuperados? |
| **Context Precision** | Os documentos recuperados são realmente relevantes? |

### Framework RAGAS

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Dados de avaliação
test_data = {
    "question": ["Quantos dias de férias tenho?"],
    "answer": ["Você tem direito a 30 dias de férias anuais."],
    "contexts": [["Política de Férias: 30 dias anuais após 12 meses"]],
    "ground_truth": ["30 dias por ano"]
}

result = evaluate(
    dataset=test_data,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(result)
```

---

## 7.5 RAG Avançado — Técnicas de Melhoria

### 1. Query Rewriting

Reescrever a pergunta do usuário para melhorar a recuperação:

```python
def rewrite_query(original_query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Reescreva esta pergunta para maximizar a recuperação 
            de documentos relevantes. Torne-a mais específica e inclua sinônimos relevantes.
            Retorne apenas a pergunta reescrita.
            
            Pergunta original: {original_query}"""
        }],
        temperature=0
    )
    return response.choices[0].message.content

# "férias" → "política de férias, dias de descanso, período de licença remunerada"
```

### 2. Multi-Query

Gerar múltiplas variações da pergunta e combinar resultados:

```python
def multi_query_search(question: str, n_queries: int = 3) -> list:
    # Gerar variações
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user", 
            "content": f"""Gere {n_queries} variações diferentes desta pergunta 
            para busca em documentos. Cada variação em uma linha.
            
            Pergunta: {question}"""
        }]
    )
    
    queries = [question] + response.choices[0].message.content.strip().split("\n")
    
    # Buscar com cada variação e combinar
    all_results = set()
    for q in queries:
        results = collection.query(query_texts=[q], n_results=3)
        for doc in results["documents"][0]:
            all_results.add(doc)
    
    return list(all_results)
```

### 3. Reranking

Após recuperar candidatos, usar um modelo de reranking para reordenar por relevância:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list[str], top_k: int = 3) -> list[str]:
    """Usa cross-encoder para reordenar documentos por relevância."""
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# Fluxo com reranking
candidates = collection.query(query_texts=[query], n_results=10)["documents"][0]
reranked = rerank(query, candidates, top_k=3)
```

### 4. Hybrid Search

Combina busca vetorial (semântica) com busca por palavras-chave (BM25):

```python
from rank_bm25 import BM25Okapi

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        # BM25 para busca léxica
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.5):
        # Busca léxica (BM25)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # Busca semântica (vetorial)
        semantic_results = collection.query(query_texts=[query], n_results=len(self.documents))
        semantic_scores = [0.0] * len(self.documents)
        # mapear scores...
        
        # Combinar (Reciprocal Rank Fusion ou média ponderada)
        combined = alpha * bm25_scores + (1 - alpha) * semantic_scores
        
        top_indices = combined.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### 5. Self-RAG (RAG Reflexivo)

O modelo avalia se precisa buscar informação antes de responder:

```python
def self_rag(question: str) -> str:
    # 1. O modelo decide se precisa de retrieval
    decision_prompt = f"""Você precisa de informações externas para responder esta pergunta?
    Responda apenas SIM ou NÃO.
    
    Pergunta: {question}"""
    
    need_retrieval = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": decision_prompt}]
    ).choices[0].message.content.strip().upper()
    
    if need_retrieval == "SIM":
        context = retrieve(question)
        return generate_with_context(question, context)
    else:
        return generate_direct(question)
```

---

## 7.6 Chunking Estratégico

A qualidade do chunking impacta muito o RAG:

### Chunking Hierárquico

```python
# Armazena tanto chunks pequenos (precisos) quanto grandes (contextuais)
# Recupera pelo chunk pequeno, envia o grande ao LLM
def hierarchical_chunk(document: str):
    # Chunks pequenos para busca precisa
    small_chunks = chunk_text(document, size=200)
    # Chunks grandes para contexto rico
    large_chunks = chunk_text(document, size=1000)
    
    return small_chunks, large_chunks
```

### Chunking Semântico

```python
# Divide o texto em pontos de mudança semântica
from semantic_text_splitter import TextSplitter

splitter = TextSplitter.from_huggingface_tokenizer("gpt2", capacity=512)
chunks = splitter.chunks(document)
```

---

## 7.7 Construindo um Chatbot RAG Completo

```python
class RAGChatbot:
    def __init__(self, collection):
        self.client = OpenAI()
        self.collection = collection
        self.history = []
    
    def chat(self, user_message: str) -> str:
        # 1. Buscar contexto
        context = self._retrieve(user_message)
        
        # 2. Construir mensagens com histórico
        messages = [
            {"role": "system", "content": f"""Você é um assistente prestativo.
Use o contexto fornecido para responder. Se não souber, diga que não sabe.

CONTEXTO RELEVANTE:
{context}"""}
        ] + self.history + [
            {"role": "user", "content": user_message}
        ]
        
        # 3. Gerar resposta
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # 4. Atualizar histórico
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        
        # 5. Manter histórico limitado
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        return answer
    
    def _retrieve(self, query: str, n: int = 3) -> str:
        results = self.collection.query(query_texts=[query], n_results=n)
        docs = results["documents"][0]
        return "\n\n".join(docs)
```

---

## 📌 Resumo da Parte 07

| Conceito | Descrição |
|----------|-----------|
| RAG | Arquitetura que combina recuperação + geração |
| Indexação | Chunking → Embedding → Banco vetorial (feita uma vez) |
| Consulta | Embed query → busca → contexto → LLM → resposta |
| Faithfulness | Resposta fiel ao contexto recuperado |
| Reranking | Reordenar candidatos com modelo mais preciso |
| Hybrid Search | Combina busca vetorial + palavras-chave |
| Query Rewriting | Reformular pergunta para melhor recuperação |

---

## 🔗 Referências

- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [RAGAS - Evaluation Framework](https://ragas.io)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex](https://docs.llamaindex.ai)
- [Advanced RAG Techniques](https://towardsdatascience.com/advanced-rag-techniques)

---

⬅️ **Anterior:** [Parte 06](./parte-06-bancos-vetoriais.md) | ➡️ **Próximo:** [Parte 08 — Agentes](./parte-08-agentes.md)
