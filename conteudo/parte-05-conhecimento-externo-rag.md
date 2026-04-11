# Parte 05 — Conhecimento Externo e RAG

> **Carga horária:** 6 horas  
> **Prática correspondente:** [Prática 03](../praticas/pratica-03-conhecimento-externo-rag.md)

---

## 5.1 O que são Embeddings?

**Embeddings** são representações numéricas de dados (texto, imagens, áudio) na forma de vetores de alta dimensão. A ideia central é simples e poderosa:

> **Dados semanticamente similares ficam próximos no espaço vetorial.**

```
"gato"   → [0.23, -0.15, 0.87, ..., 0.42]  (1536 dimensões)
"felino" → [0.24, -0.14, 0.86, ..., 0.44]  (próximo!)
"carro"  → [-0.78, 0.92, -0.11, ..., 0.33] (distante)
```

Esse princípio — objetos similares têm vetores próximos — é a base de:
- Busca semântica
- RAG (Retrieval Augmented Generation)
- Sistemas de recomendação
- Detecção de duplicatas e plágio
- Clustering de documentos

---

## 5.2 Como Embeddings São Gerados?

Modelos de embedding são redes neurais treinadas para mapear texto para vetores. O treinamento tipicamente usa:

### Modelos Contrastivos (SBERT, etc.)

Treinados com pares de frases similares/dissimilares:
- **Pares positivos**: "banco de dados" e "SGBD" → vetores próximos
- **Pares negativos**: "banco de dados" e "banco do Brasil" → vetores distantes

### Modelos de Linguagem Mascarados (BERT-based)

Extraem a representação do token `[CLS]` após o processamento completo do texto.

### Modelos de Embedding Dedicados

- **OpenAI text-embedding-3**: treinados especificamente para embedding
- **Sentence-Transformers**: família open-source altamente eficiente
- **E5, BGE, Nomic**: modelos open-source de alta performance

---

## 5.3 Dimensionalidade

Modelos populares e suas dimensões:

| Modelo | Dimensões | Contexto | Open-source |
|--------|-----------|----------|-------------|
| text-embedding-ada-002 | 1536 | 8191 tokens | ❌ |
| text-embedding-3-small | 1536 | 8191 tokens | ❌ |
| text-embedding-3-large | 3072 | 8191 tokens | ❌ |
| all-MiniLM-L6-v2 | 384 | 512 tokens | ✅ |
| all-mpnet-base-v2 | 768 | 514 tokens | ✅ |
| nomic-embed-text | 768 | 8192 tokens | ✅ |
| mxbai-embed-large | 1024 | 512 tokens | ✅ |

**Trade-off**: mais dimensões → mais expressividade, mas mais memória e custo de computação.

---

## 5.4 Gerando Embeddings na Prática

### Com a API da OpenAI

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Gera embedding para um texto."""
    text = text.replace("\n", " ")  # boa prática
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Exemplo
embedding = get_embedding("Inteligência Artificial Generativa")
print(f"Dimensões: {len(embedding)}")  # 1536
print(f"Primeiros valores: {embedding[:5]}")
```

### Em Lote (Batch)

```python
def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Gera embeddings para múltiplos textos de uma vez."""
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

# Muito mais eficiente que chamar uma por uma
docs = ["Python é uma linguagem", "IA está em todo lugar", "Machine learning é poderoso"]
embeddings = get_embeddings_batch(docs)
print(f"Gerados {len(embeddings)} embeddings")
```

### Com Sentence-Transformers (Gratuito, Local)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python é ótimo para ciência de dados",
    "A linguagem Python é muito usada em IA",
    "Futebol é um esporte popular",
]

embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

### Embeddings Multilinguais

Para aplicações em português:

```python
# Modelos com bom suporte a português
models_ptbr = [
    "intfloat/multilingual-e5-large",   # bom para PT-BR
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "neuralmind/bert-base-portuguese-cased",  # apenas português
]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large")

# Para e5, adicionar prefixo "query: " para consultas
query = "query: O que é inteligência artificial?"
doc = "passage: Inteligência artificial é a simulação da inteligência humana por máquinas."

q_emb = model.encode(query)
d_emb = model.encode(doc)
print(cosine_similarity(q_emb, d_emb))  # alta similaridade
```

### Fine-tuning de Embeddings

Para domínios específicos (jurídico, médico, técnico), você pode ajustar modelos de embedding:

1. **Colete pares relevantes**: (query, documento_relevante, documento_irrelevante)
2. **Use loss contrastiva**: InfoNCE, Triplet Loss, MNRL
3. **Frameworks**: `sentence-transformers`, `FlagEmbedding`

---

## 5.5 Similaridade entre Vetores

### Similaridade de Cosseno

A métrica mais comum para comparar embeddings de texto:

```python
import numpy as np

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Retorna valor entre -1 e 1 (1 = idênticos, 0 = perpendiculares, -1 = opostos)."""
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Exemplo
emb_gato   = get_embedding("gato")
emb_felino = get_embedding("felino")
emb_carro  = get_embedding("automóvel")
emb_ia     = get_embedding("inteligência artificial")

print(cosine_similarity(emb_gato, emb_felino))  # ~0.92 (muito similar)
print(cosine_similarity(emb_gato, emb_carro))   # ~0.35 (pouco similar)
print(cosine_similarity(emb_gato, emb_ia))      # ~0.15 (muito diferente)
```

### Distância Euclidiana

```python
def euclidean_distance(vec_a, vec_b) -> float:
    return float(np.linalg.norm(np.array(vec_a) - np.array(vec_b)))
```

### Produto Interno (Dot Product)

Para vetores normalizados (unitários), dot product = similaridade de cosseno. Muitos bancos vetoriais usam esta métrica por ser mais rápida de computar.

### Busca Semântica

A aplicação mais direta de embeddings:

```python
def semantic_search(query: str, documents: list[str], top_k: int = 3):
    """Encontra os documentos mais relevantes para uma consulta."""
    
    # 1. Gera embeddings para todos os documentos
    doc_embeddings = get_embeddings_batch(documents)
    
    # 2. Gera embedding da consulta
    query_embedding = get_embedding(query)
    
    # 3. Calcula similaridade com cada documento
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    
    # 4. Ordena por similaridade e retorna top-k
    ranked = sorted(
        zip(similarities, documents),
        key=lambda x: x[0],
        reverse=True
    )
    
    return ranked[:top_k]

# Uso
documents = [
    "Python é uma linguagem de programação de alto nível",
    "Machine Learning é um subcampo da IA",
    "O Brasil tem 215 milhões de habitantes",
    "Redes neurais aprendem padrões de dados",
    "A Copa do Mundo é realizada a cada 4 anos",
]

results = semantic_search("Como funciona aprendizado de máquina?", documents)
for score, doc in results:
    print(f"{score:.3f} | {doc}")
```

**Saída esperada:**
```
0.847 | Machine Learning é um subcampo da IA
0.831 | Redes neurais aprendem padrões de dados
0.614 | Python é uma linguagem de programação de alto nível
```

---

## 5.6 Chunking de Documentos

Documentos longos precisam ser divididos em pedaços (chunks) antes de gerar embeddings:

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Divide texto em chunks com sobreposição para preservar contexto.
    
    chunk_size: tamanho em caracteres de cada chunk
    overlap: sobreposição entre chunks consecutivos
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Tenta quebrar em limite de palavra/parágrafo
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_size * 0.7:  # não muito pequeno
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if len(c) > 50]  # remove chunks muito pequenos

# Chunking por parágrafo (geralmente melhor)
def chunk_by_paragraph(text: str, max_chars: int = 1000) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

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

## 5.7 Por que Bancos Vetoriais?

Você já sabe gerar embeddings e calcular similaridade. Mas e quando você tem:
- 1 milhão de documentos?
- 100 usuários simultâneos fazendo buscas?
- Necessidade de atualizar documentos?

Fazer busca por força bruta (comparar a query com todos os vetores) fica inviável em escala. Os **bancos de dados vetoriais** resolvem isso.

> **Definição:** Um banco de dados vetorial é um sistema otimizado para armazenar, indexar e buscar vetores de alta dimensão de forma eficiente e escalável.

---

## 5.8 Busca Aproximada de Vizinhos (ANN)

Em vez de busca exata (comparar com todos), os bancos vetoriais usam algoritmos de **busca aproximada (ANN - Approximate Nearest Neighbor)**:

### HNSW (Hierarchical Navigable Small World)

O algoritmo mais popular:
- Cria um grafo hierárquico de vizinhança
- Navega do nível mais alto (sparse) ao mais baixo (denso)
- Muito eficiente para alta dimensionalidade

### IVF (Inverted File Index)

- Clusteriza vetores em grupos (Voronoi cells)
- Na busca, examina apenas os clusters mais próximos
- Bom para datasets muito grandes

### ScaNN, Annoy, FAISS

Outros algoritmos com diferentes trade-offs entre velocidade, precisão e memória.

---

## 5.9 Principais Bancos Vetoriais

### Comparativo

| Banco | Tipo | Hospedagem | Destaques |
|-------|------|-----------|-----------|
| **ChromaDB** | Open-source | Local / Cloud | Simples, ideal para dev |
| **FAISS** | Biblioteca | Local | Ultra-rápido, Meta |
| **Pinecone** | SaaS | Cloud | Gerenciado, escalável |
| **Weaviate** | Open-source | Local / Cloud | Schema flexível, GraphQL |
| **Qdrant** | Open-source | Local / Cloud | Alta performance, filtros |
| **Milvus** | Open-source | Local / Cloud | Enterprise, Kubernetes |
| **pgvector** | Extensão | PostgreSQL | Se já usa PostgreSQL |

### FAISS — Alta Performance Local

```bash
pip install faiss-cpu  # CPU
# pip install faiss-gpu  # GPU (requer CUDA)
```

```python
import faiss
import numpy as np

# Simulando embeddings de 1536 dimensões
dimension = 1536
n_docs = 10000

# Criar índice HNSW
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = número de vizinhos no grafo

# Adicionar vetores (deve ser float32)
vectors = np.random.rand(n_docs, dimension).astype(np.float32)
faiss.normalize_L2(vectors)  # normalizar para similaridade de cosseno
index.add(vectors)

# Buscar
query = np.random.rand(1, dimension).astype(np.float32)
faiss.normalize_L2(query)

distances, indices = index.search(query, k=5)
print(f"Índices mais próximos: {indices[0]}")
print(f"Distâncias: {distances[0]}")

# Salvar e carregar
faiss.write_index(index, "meu_index.faiss")
index = faiss.read_index("meu_index.faiss")
```

### pgvector — Vetores no PostgreSQL

Para quem já usa PostgreSQL, pgvector adiciona suporte nativo a vetores:

```sql
-- Habilitar extensão
CREATE EXTENSION IF NOT EXISTS vector;

-- Criar tabela com coluna vetorial
CREATE TABLE documentos (
    id SERIAL PRIMARY KEY,
    conteudo TEXT,
    embedding vector(1536),
    metadata JSONB,
    criado_em TIMESTAMP DEFAULT NOW()
);

-- Criar índice HNSW
CREATE INDEX ON documentos USING hnsw (embedding vector_cosine_ops);

-- Inserir documento com embedding (em Python)
-- INSERT INTO documentos (conteudo, embedding) VALUES ($1, $2::vector)

-- Buscar os 5 mais similares
SELECT id, conteudo, 1 - (embedding <=> query_embedding) AS similaridade
FROM documentos
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

```python
# Python com psycopg2
import psycopg2
import json

conn = psycopg2.connect("postgresql://user:pass@localhost/db")
cur = conn.cursor()

# Inserir
embedding = get_embedding("texto do documento")
cur.execute(
    "INSERT INTO documentos (conteudo, embedding) VALUES (%s, %s)",
    ("texto do documento", embedding)
)

# Buscar
query_emb = get_embedding("minha consulta")
cur.execute("""
    SELECT conteudo, 1 - (embedding <=> %s::vector) AS score
    FROM documentos
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_emb, query_emb))

for row in cur.fetchall():
    print(f"{row[1]:.3f} | {row[0]}")
```

### Filtragem com Metadados

Um diferencial importante dos bancos vetoriais é combinar **busca vetorial + filtros de metadados**:

```python
# ChromaDB com filtros
results = collection.query(
    query_texts=["como aprender machine learning?"],
    n_results=3,
    where={"categoria": "ia"}  # filtro de metadado
)

# Filtros mais complexos
results = collection.query(
    query_texts=["consulta"],
    where={
        "$and": [
            {"categoria": {"$in": ["ia", "dados"]}},
            {"dificuldade": {"$ne": "alta"}}
        ]
    }
)
```

### Boas Práticas para Bancos Vetoriais

| Prática | Motivo |
|---------|--------|
| Normalizar vetores | Garante consistência na similaridade de cosseno |
| Chunk cuidadoso | Chunks muito pequenos perdem contexto; muito grandes perdem precisão |
| Metadados ricos | Permitem filtragem eficiente sem vetorização |
| Monitorar qualidade | Avalie a relevância dos resultados periodicamente |
| Backup regular | Especialmente para bancos persistentes |
| Índice correto | HNSW para alta precisão; IVF para datasets gigantes |

---

## 5.10 ChromaDB — Início Rápido

ChromaDB é a escolha ideal para aprender e prototipar:

```bash
pip install chromadb
```

### Operações Básicas

```python
import chromadb

# Iniciar cliente (dados em memória)
client = chromadb.Client()

# Para persistir em disco:
client = chromadb.PersistentClient(path="./meu_banco")

# Criar uma coleção
collection = client.create_collection(
    name="documentos",
    metadata={"hnsw:space": "cosine"}  # métrica de distância
)

# Adicionar documentos
collection.add(
    documents=[
        "Python é uma linguagem de alto nível",
        "Machine Learning é uma área da IA",
        "JavaScript é usado para web",
        "RAG combina recuperação com geração",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],
    metadatas=[
        {"categoria": "linguagem", "dificuldade": "baixa"},
        {"categoria": "ia", "dificuldade": "alta"},
        {"categoria": "linguagem", "dificuldade": "media"},
        {"categoria": "ia", "dificuldade": "alta"},
    ]
)

print(f"Total de documentos: {collection.count()}")
```

### Busca Semântica

```python
# ChromaDB gera embeddings automaticamente (usa modelo embutido)
results = collection.query(
    query_texts=["como aprender inteligência artificial?"],
    n_results=2
)

for doc, score, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"[{score:.3f}] {doc} | {meta}")
```

### Usando Embeddings Próprios

```python
import chromadb
from openai import OpenAI

openai_client = OpenAI()

def embed(texts):
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

# Configurar ChromaDB com função de embedding customizada
class OpenAIEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input):
        return embed(input)

collection = client.create_collection(
    name="docs_openai",
    embedding_function=OpenAIEmbeddingFunction()
)
```

### Padrões de Uso

#### Padrão: Index-then-Query

```python
# FASE 1: Indexação (feita uma vez ou periodicamente)
def index_documents(documents: list[dict]):
    texts = [doc["content"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    collection.add(documents=texts, ids=ids, metadatas=metadatas)

# FASE 2: Consulta (feita por cada usuário/request)
def search(query: str, filters: dict = None, top_k: int = 5):
    return collection.query(
        query_texts=[query],
        n_results=top_k,
        where=filters
    )
```

#### Padrão: Upsert (atualizar ou inserir)

```python
# Evita duplicatas ao reindexar documentos atualizados
collection.upsert(
    documents=["novo conteúdo do documento"],
    ids=["doc1"],  # se já existe, atualiza
    metadatas=[{"versao": 2}]
)
```

---

## 5.11 O que é RAG?

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

## 5.12 Arquitetura RAG

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

### Avaliação de RAG

#### Métricas Principais

| Métrica | Descrição |
|---------|-----------|
| **Faithfulness** | A resposta é fiel ao contexto recuperado? |
| **Answer Relevancy** | A resposta é relevante para a pergunta? |
| **Context Recall** | Os documentos relevantes foram recuperados? |
| **Context Precision** | Os documentos recuperados são realmente relevantes? |

#### Framework RAGAS

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

## 5.13 RAG Simples — Implementação do Zero

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

## 5.14 RAG Avançado — Técnicas de Melhoria

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

## 5.15 Construindo um Chatbot RAG Completo

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

## 📌 Resumo da Parte 05

| Conceito | Descrição |
|----------|-----------|
| Embedding | Vetor numérico que representa semanticamente um texto |
| Similaridade de Cosseno | Mede ângulo entre vetores (0-1 para texto) |
| Chunking | Dividir documentos longos antes de embedar |
| Busca Semântica | Encontrar documentos por significado, não palavras-chave |
| Modelo de Embedding | Rede neural que transforma texto em vetor |
| Dimensões | Tamanho do vetor — mais dimensões = mais capacidade |
| Banco Vetorial | BD otimizado para armazenar e buscar vetores |
| ANN | Busca aproximada de vizinhos mais próximos |
| HNSW | Algoritmo de indexação baseado em grafos hierárquicos |
| ChromaDB | Banco vetorial simples para dev e prototipagem |
| FAISS | Biblioteca de alta performance da Meta |
| pgvector | Extensão para adicionar vetores ao PostgreSQL |
| Filtragem híbrida | Combina busca vetorial com filtros de metadados |
| RAG | Arquitetura que combina recuperação + geração |
| Indexação | Chunking → Embedding → Banco vetorial (feita uma vez) |
| Consulta | Embed query → busca → contexto → LLM → resposta |
| Faithfulness | Resposta fiel ao contexto recuperado |
| Reranking | Reordenar candidatos com modelo mais preciso |
| Hybrid Search | Combina busca vetorial + palavras-chave |
| Query Rewriting | Reformular pergunta para melhor recuperação |

---

## 🔗 Referências

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard) — ranking de modelos de embedding
- [Understanding Embeddings](https://simonwillison.net/2023/Oct/23/embeddings/)
- [ChromaDB Docs](https://docs.trychroma.com)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Qdrant Docs](https://qdrant.tech/documentation)
- [Vector Database Comparison](https://ann-benchmarks.com)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [RAGAS - Evaluation Framework](https://ragas.io)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex](https://docs.llamaindex.ai)
- [Advanced RAG Techniques](https://towardsdatascience.com/advanced-rag-techniques)

---

⬅️ **Anterior:** [Parte 04 — Engenharia de Contexto II](./parte-04-engenharia-contexto-ii.md) | ➡️ **Próximo:** [Parte 06 — Agentes no Sistema](./parte-06-agentes-no-sistema.md)
