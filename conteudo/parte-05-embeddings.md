# Parte 05 — Embeddings e Representação Vetorial

> **Carga horária:** 3 horas  
> **Prática correspondente:** [Prática 03](../praticas/pratica-03-embeddings.md)

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

---

## 5.6 Busca Semântica

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

## 5.7 Chunking de Documentos

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

---

## 5.8 Embeddings Multilinguais

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

---

## 5.9 Fine-tuning de Embeddings

Para domínios específicos (jurídico, médico, técnico), você pode ajustar modelos de embedding:

1. **Colete pares relevantes**: (query, documento_relevante, documento_irrelevante)
2. **Use loss contrastiva**: InfoNCE, Triplet Loss, MNRL
3. **Frameworks**: `sentence-transformers`, `FlagEmbedding`

---

## 📌 Resumo da Parte 05

| Conceito | Descrição |
|----------|-----------|
| Embedding | Vetor numérico que representa semânticamente um texto |
| Similaridade de Cosseno | Mede ângulo entre vetores (0-1 para texto) |
| Chunking | Dividir documentos longos antes de embedar |
| Busca Semântica | Encontrar documentos por significado, não palavras-chave |
| Modelo de Embedding | Rede neural que transforma texto em vetor |
| Dimensões | Tamanho do vetor — mais dimensões = mais capacidade |

---

## 🔗 Referências

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard) — ranking de modelos de embedding
- [Understanding Embeddings](https://simonwillison.net/2023/Oct/23/embeddings/)

---

⬅️ **Anterior:** [Parte 04](./parte-04-prompt-engineering.md) | ➡️ **Próximo:** [Parte 06 — Bancos Vetoriais](./parte-06-bancos-vetoriais.md)
