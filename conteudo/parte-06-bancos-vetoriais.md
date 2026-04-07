# Parte 06 — Bancos de Dados Vetoriais

> **Carga horária:** 2 horas  
> **Prática correspondente:** [Prática 04](../praticas/pratica-04-banco-vetorial.md)

---

## 6.1 Por que Bancos Vetoriais?

Você já sabe gerar embeddings e calcular similaridade. Mas e quando você tem:
- 1 milhão de documentos?
- 100 usuários simultâneos fazendo buscas?
- Necessidade de atualizar documentos?

Fazer busca por força bruta (comparar a query com todos os vetores) fica inviável em escala. Os **bancos de dados vetoriais** resolvem isso.

> **Definição:** Um banco de dados vetorial é um sistema otimizado para armazenar, indexar e buscar vetores de alta dimensão de forma eficiente e escalável.

---

## 6.2 Busca Aproximada de Vizinhos Mais Próximos (ANN)

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

## 6.3 Principais Bancos Vetoriais

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

---

## 6.4 ChromaDB — Início Rápido

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

---

## 6.5 FAISS — Alta Performance Local

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

---

## 6.6 pgvector — Vetores no PostgreSQL

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

---

## 6.7 Filtragem com Metadados

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

---

## 6.8 Padrões de Uso

### Padrão: Index-then-Query

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

### Padrão: Upsert (atualizar ou inserir)

```python
# Evita duplicatas ao reindexar documentos atualizados
collection.upsert(
    documents=["novo conteúdo do documento"],
    ids=["doc1"],  # se já existe, atualiza
    metadatas=[{"versao": 2}]
)
```

---

## 6.9 Boas Práticas

| Prática | Motivo |
|---------|--------|
| Normalizar vetores | Garante consistência na similaridade de cosseno |
| Chunk cuidadoso | Chunks muito pequenos perdem contexto; muito grandes perdem precisão |
| Metadados ricos | Permitem filtragem eficiente sem vetorização |
| Monitorar qualidade | Avalie a relevância dos resultados periodicamente |
| Backup regular | Especialmente para bancos persistentes |
| Índice correto | HNSW para alta precisão; IVF para datasets gigantes |

---

## 📌 Resumo da Parte 06

| Conceito | Descrição |
|----------|-----------|
| Banco Vetorial | BD otimizado para armazenar e buscar vetores |
| ANN | Busca aproximada de vizinhos mais próximos |
| HNSW | Algoritmo de indexação baseado em grafos hierárquicos |
| ChromaDB | Banco vetorial simples para dev e prototipagem |
| FAISS | Biblioteca de alta performance da Meta |
| pgvector | Extensão para adicionar vetores ao PostgreSQL |
| Filtragem híbrida | Combina busca vetorial com filtros de metadados |

---

## 🔗 Referências

- [ChromaDB Docs](https://docs.trychroma.com)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Qdrant Docs](https://qdrant.tech/documentation)
- [Vector Database Comparison](https://ann-benchmarks.com)

---

⬅️ **Anterior:** [Parte 05](./parte-05-embeddings.md) | ➡️ **Próximo:** [Parte 07 — RAG](./parte-07-rag.md)
