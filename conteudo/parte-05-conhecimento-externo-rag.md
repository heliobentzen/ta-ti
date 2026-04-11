# Parte 05 — Conhecimento Externo e RAG

> **Carga horária:** 6h  
> **Prática correspondente:** [Prática 05](../praticas/pratica-05-rag.md)

---

## 5.1 Por que RAG? O problema de conhecimento em LLMs

LLMs são treinados em snapshots do passado. O GPT-4 tem cutoff em abril de 2023. O Claude tem o seu. Qualquer modelo que você usa hoje não sabe o que aconteceu na semana passada — e mais importante: não sabe nada sobre os documentos internos da sua empresa, a documentação do seu sistema legado, as regras de negócio do seu domínio.

Isso cria três problemas reais em produção:

1. **Conhecimento desatualizado** — o modelo não sabe sobre mudanças recentes em APIs, legislação, preços, etc.
2. **Alucinação por falta de contexto** — sem informação real, o modelo inventa respostas plausíveis mas incorretas.
3. **Falta de conhecimento de domínio** — documentação técnica, manuais de equipamento, jurisprudência interna, políticas de RH.

### Fine-tuning vs. RAG vs. Prompt engineering

Esta é uma decisão que aparece em todo projeto. A tabela abaixo é brutal mas honesta:

| Abordagem | Quando usar | Custo | Atualização | Transparência |
|-----------|-------------|-------|-------------|---------------|
| **Prompt engineering** | Comportamento geral, personalidade, estilo | Baixo | Imediata | Alta |
| **RAG** | Conhecimento factual específico, docs internos | Médio | Fácil (reindexar) | Alta (fonte rastreável) |
| **Fine-tuning** | Estilo muito específico, formato de saída, domínio com vocabulário único | Alto | Custoso (re-treinar) | Baixa |

**A regra prática:** se o problema é "o modelo não sabe X fato", use RAG. Se o problema é "o modelo não responde no formato/estilo correto", considere fine-tuning. Se o problema é "o modelo não sabe como se comportar", use prompt engineering.

Fine-tuning ensina ao modelo *como pensar*, não *o que saber*. Essa distinção elimina 80% das dúvidas.

### Quando RAG é overkill

RAG adiciona complexidade operacional real. Não use se:

- Você tem poucos documentos que cabem no contexto do modelo (< 50 páginas)
- Os documentos mudam raramente e o conteúdo pode ser embutido no system prompt
- A latência extra de retrieval é inaceitável para o seu caso de uso
- O volume de queries é baixo e o custo de contexto grande é aceitável

### O pipeline RAG em visão geral

```
[Documentos] → [Extração] → [Chunking] → [Embedding] → [Índice Vetorial]
                                                               ↓
[Usuário] → [Query] → [Embedding da query] → [Busca no índice] → [Top-K chunks]
                                                                        ↓
                                               [LLM com chunks no contexto] → [Resposta]
```

Cada seta é um ponto de falha. Cada etapa tem trade-offs. Vamos destrinchar cada uma.

---

## 5.2 Ingestion: coleta e processamento de documentos

O lixo entra, lixo sai. A qualidade do seu pipeline de ingestion determina o teto da qualidade do seu RAG. Muito time subestima esta etapa.

### Formatos de documento e seus desafios

| Formato | Biblioteca Python | Desafio principal |
|---------|------------------|-------------------|
| PDF | `pymupdf`, `pdfplumber`, `pypdf` | Tabelas, colunas múltiplas, PDFs escaneados (precisa OCR) |
| HTML | `beautifulsoup4`, `trafilatura` | Remover nav, footer, ads; preservar estrutura |
| DOCX | `python-docx` | Tabelas embutidas, imagens com texto |
| Markdown | Direto (é texto) | Praticamente nenhum |
| Banco de dados | SQLAlchemy | Definir quais campos são relevantes |
| Emails | `mailparser` | Threading, anexos, assinaturas |

### Pipeline de ingestion com tratamento de erros

```python
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf  # pip install pymupdf
from bs4 import BeautifulSoup  # pip install beautifulsoup4

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


def extract_pdf(path: Path) -> Optional[Document]:
    try:
        doc = pymupdf.open(str(path))
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

        if not pages:
            logger.warning(f"PDF vazio ou escaneado (sem texto extraível): {path}")
            return None

        content = "\n\n".join(pages)
        return Document(
            content=content,
            source=str(path),
            doc_type="pdf",
            metadata={"pages": len(pages), "filename": path.name},
        )
    except Exception as e:
        logger.error(f"Falha ao extrair PDF {path}: {e}")
        return None


def extract_html(path: Path) -> Optional[Document]:
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Remove elementos não-conteúdo
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        content = "\n".join(lines)

        title = soup.title.string if soup.title else path.stem
        return Document(
            content=content,
            source=str(path),
            doc_type="html",
            metadata={"title": title, "filename": path.name},
        )
    except Exception as e:
        logger.error(f"Falha ao extrair HTML {path}: {e}")
        return None


def extract_markdown(path: Path) -> Optional[Document]:
    try:
        content = path.read_text(encoding="utf-8")
        return Document(
            content=content,
            source=str(path),
            doc_type="markdown",
            metadata={"filename": path.name},
        )
    except Exception as e:
        logger.error(f"Falha ao ler Markdown {path}: {e}")
        return None


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_markdown,
    ".txt": extract_markdown,
}


def ingest_directory(directory: Path) -> list[Document]:
    documents = []
    skipped = 0

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        extractor = EXTRACTORS.get(path.suffix.lower())
        if extractor is None:
            skipped += 1
            continue
        doc = extractor(path)
        if doc:
            documents.append(doc)

    logger.info(
        f"Ingestion: {len(documents)} documentos extraídos, "
        f"{skipped} arquivos ignorados (formato não suportado)"
    )
    return documents
```

### Limpeza e pré-processamento

```python
import re


def clean_text(text: str) -> str:
    # Remove caracteres de controle (exceto newlines e tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normaliza quebras de linha
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)

    # Colapsa mais de 2 linhas em branco consecutivas
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [l.rstrip() for l in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def is_content_too_short(text: str, min_chars: int = 100) -> bool:
    """Descarta documentos que provavelmente são artefatos (headers, páginas em branco)."""
    return len(text.strip()) < min_chars
```

---

## 5.3 Chunking: dividindo documentos para recuperação

Um documento de 50 páginas não pode ser recuperado como uma unidade — você não quer trazer 50 páginas para o contexto quando a pergunta só precisa de 2 parágrafos. Chunking é o processo de dividir documentos em pedaços recuperáveis.

### Por que chunking importa: precisão de recuperação

A granularidade do chunk determina dois trade-offs opostos:

- **Chunk muito pequeno**: retrieval preciso, mas sem contexto suficiente para o modelo responder bem. Pior: pode dividir no meio de uma explicação.
- **Chunk muito grande**: mais contexto, mas você traz ruído junto. O modelo pode ter dificuldade em focar na parte relevante.

O tamanho ideal depende do domínio e do tipo de pergunta. Para FAQs técnicas, chunks de 256–512 tokens costumam funcionar bem. Para documentos legais ou científicos, 512–1024 tokens pode ser necessário.

### Estratégias de chunking

```python
import re


def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Divide por número de caracteres com overlap.
    Simples e previsível. Bom ponto de partida.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(
    text: str,
    max_chunk_size: int = 1000,
    overlap_paragraphs: int = 1,
) -> list[str]:
    """
    Divide por parágrafos (linhas em branco).
    Respeita estrutura semântica do texto.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current_chunk_paragraphs = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if current_size + para_size > max_chunk_size and current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))
            # Overlap: mantém últimos N parágrafos
            current_chunk_paragraphs = current_chunk_paragraphs[-overlap_paragraphs:]
            current_size = sum(len(p) for p in current_chunk_paragraphs)

        current_chunk_paragraphs.append(para)
        current_size += para_size

    if current_chunk_paragraphs:
        chunks.append("\n\n".join(current_chunk_paragraphs))

    return chunks


def chunk_by_sentence(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    """Divide por sentenças. Bom para textos contínuos."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    step = sentences_per_chunk - overlap_sentences
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))

    return chunks


def chunk_markdown_by_heading(
    text: str,
    max_chunk_size: int = 1500,
) -> list[dict]:
    """
    Divide Markdown por cabeçalhos.
    Preserva estrutura hierárquica e retorna metadados de seção.
    """
    lines = text.splitlines()
    chunks = []
    current_section = {"heading": "", "level": 0, "content": []}

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            if current_section["content"]:
                content = "\n".join(current_section["content"]).strip()
                if content:
                    chunks.append({
                        "content": content,
                        "heading": current_section["heading"],
                        "heading_level": current_section["level"],
                    })
            level = len(heading_match.group(1))
            heading = heading_match.group(2)
            current_section = {"heading": heading, "level": level, "content": [line]}
        else:
            current_section["content"].append(line)

    if current_section["content"]:
        content = "\n".join(current_section["content"]).strip()
        if content:
            chunks.append({
                "content": content,
                "heading": current_section["heading"],
                "heading_level": current_section["level"],
            })

    # Chunks grandes demais são subdivididos
    final_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > max_chunk_size:
            sub_chunks = chunk_by_paragraph(chunk["content"], max_chunk_size)
            for sub in sub_chunks:
                final_chunks.append({**chunk, "content": sub})
        else:
            final_chunks.append(chunk)

    return final_chunks
```

### O trade-off de tamanho de chunk

| Tamanho | Tokens aprox. | Vantagem | Desvantagem |
|---------|---------------|----------|-------------|
| Pequeno | 128–256 | Alta precisão de retrieval | Contexto insuficiente, fragmentação |
| Médio | 512–768 | Equilíbrio | Ponto de partida recomendado |
| Grande | 1024–2048 | Contexto rico | Ruído, custo de embedding maior |

**Dica de produção:** comece com 512 tokens e overlap de 10%. Meça a qualidade do retrieval (Seção 5.7) e ajuste baseado em dados reais, não em intuição.

---

## 5.4 Embeddings: escolha do modelo

Embedding é a representação vetorial do texto. A qualidade desse vetor determina se a busca semântica vai funcionar.

### Critérios de escolha

| Critério | O que considerar |
|----------|-----------------|
| **Qualidade** | Performance em benchmarks (MTEB) para o seu idioma e domínio |
| **Idioma** | Modelos multilíngues vs. específicos para PT-BR |
| **Dimensão** | Mais dimensões = mais qualidade (geralmente), mas mais custo de armazenamento |
| **Custo** | Por token (comercial) vs. custo de infraestrutura (self-hosted) |
| **Latência** | Crítico se o embedding acontece em tempo real na query |

### Modelos disponíveis

| Modelo | Tipo | Dimensão | Notas PT-BR |
|--------|------|----------|-------------|
| `text-embedding-3-small` | OpenAI (pago) | 1536 | Bom, mas pago por token |
| `text-embedding-3-large` | OpenAI (pago) | 3072 | Melhor qualidade, mais caro |
| `all-MiniLM-L6-v2` | Open-source | 384 | Leve, foco em inglês |
| `nomic-embed-text-v1` | Open-source | 768 | Multilíngue, boa qualidade |
| `BAAI/bge-m3` | Open-source | 1024 | Excelente multilíngue, inclui PT-BR |
| `intfloat/multilingual-e5-large` | Open-source | 1024 | Forte em PT-BR |

Para projetos em Português, `BAAI/bge-m3` ou `multilingual-e5-large` são as escolhas mais seguras no cenário open-source.

### Implementação com caching

```python
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache simples em disco para embeddings. Evita reprocessar o mesmo texto."""

    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key(self, text: str, model: str) -> str:
        payload = f"{model}:{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[list[float]]:
        path = self.cache_dir / f"{self._key(text, model)}.json"
        if path.exists():
            self._hits += 1
            return json.loads(path.read_text())
        self._misses += 1
        return None

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        path = self.cache_dir / f"{self._key(text, model)}.json"
        path.write_text(json.dumps(embedding))

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_openai: bool = False,
        cache_dir: str = ".embedding_cache",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.use_openai = use_openai
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_dir)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.use_openai:
            from openai import OpenAI
            self._model = OpenAI()
        else:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Carregando modelo {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Modelo carregado.")

    def embed_text(self, text: str) -> list[float]:
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            return cached

        self._load_model()

        if self.use_openai:
            response = self._model.embeddings.create(
                input=text,
                model=self.model_name,
            )
            embedding = response.data[0].embedding
        else:
            embedding = self._model.encode(text, normalize_embeddings=True).tolist()

        self.cache.set(text, self.model_name, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = [None] * len(texts)
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results

        self._load_model()
        uncached_texts = [texts[i] for i in uncached_indices]

        new_embeddings = []
        for start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[start : start + self.batch_size]
            if self.use_openai:
                response = self._model.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                batch_embeddings = [d.embedding for d in response.data]
            else:
                batch_embeddings = self._model.encode(
                    batch, normalize_embeddings=True
                ).tolist()
            new_embeddings.extend(batch_embeddings)

        for idx, embedding in zip(uncached_indices, new_embeddings):
            self.cache.set(texts[idx], self.model_name, embedding)
            results[idx] = embedding

        logger.info(
            f"Cache hit rate: {self.cache.hit_rate:.1%} "
            f"({self.cache._hits} hits, {self.cache._misses} misses)"
        )
        return results
```

---

## 5.5 Bancos de dados vetoriais

O banco de dados vetorial armazena seus chunks e seus embeddings, e permite busca por similaridade.

### Opções e quando usar cada uma

| Banco | Tipo | Escala | Complexidade operacional | Filtragem de metadados |
|-------|------|--------|--------------------------|------------------------|
| **ChromaDB** | Self-hosted / embarcado | Pequena-média | Baixa (Python puro) | Sim, básica |
| **Qdrant** | Self-hosted / cloud | Média-grande | Média | Excelente |
| **FAISS** | Biblioteca (sem servidor) | Qualquer (memória) | Baixa | Não nativa |
| **Pinecone** | Cloud gerenciado | Qualquer | Baixa (SaaS) | Sim |
| **Weaviate** | Self-hosted / cloud | Média-grande | Alta | Excelente |
| **pgvector** | PostgreSQL extensão | Média | Média | SQL completo |

**Para projetos iniciantes e médios:** ChromaDB é a escolha certa. Zero infra, Python puro, funciona localmente.  
**Para produção séria:** Qdrant (self-hosted) ou Pinecone (managed). pgvector se você já usa Postgres.

### Filtragem de metadados: tão importante quanto busca semântica

Em produção, raramente você busca em *todos* os documentos. Você busca em documentos de um cliente específico, de um período, de uma categoria. Metadados bem planejados são a diferença entre um RAG genérico e um RAG útil.

### ChromaDB: operações CRUD com metadados

```python
import chromadb  # pip install chromadb
from chromadb.config import Settings
import uuid


def create_chroma_client(persist_dir: str = ".chroma_db") -> chromadb.Client:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.Client,
    name: str,
    embedding_function=None,
) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    collection: chromadb.Collection,
    chunks: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
    ids: list[str] = None,
) -> None:
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in chunks]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"{len(chunks)} chunks indexados.")


def search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 5,
    where: dict = None,
) -> list[dict]:
    """
    Exemplo de where para filtragem:
      where={"source": "manual_produto_v2.pdf"}
      where={"$and": [{"category": "legal"}, {"year": {"$gte": 2023}}]}
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
            "score": 1 - dist,
        })

    return chunks


def delete_by_source(collection: chromadb.Collection, source: str) -> None:
    results = collection.get(where={"source": source})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Removidos {len(results['ids'])} chunks de '{source}'")


# Exemplo de uso
if __name__ == "__main__":
    client = create_chroma_client()
    collection = get_or_create_collection(client, "documentos_empresa")

    chunks = [
        "A política de férias permite 30 dias corridos por ano.",
        "Benefícios incluem plano de saúde e vale-refeição de R$ 35/dia.",
        "O período de experiência é de 90 dias conforme CLT.",
    ]
    metadatas = [
        {"source": "rh_politicas.pdf", "category": "rh", "year": 2024},
        {"source": "rh_beneficios.pdf", "category": "rh", "year": 2024},
        {"source": "contratos_modelo.pdf", "category": "juridico", "year": 2023},
    ]
    # Embeddings reais seriam gerados pelo EmbeddingService
    embeddings = [[0.1] * 384] * 3

    index_chunks(collection, chunks, metadatas, embeddings)

    query_emb = [0.1] * 384
    results = search(
        collection,
        query_emb,
        n_results=3,
        where={"category": "rh"},
    )
    for r in results:
        print(f"Score: {r['score']:.3f} | {r['content'][:80]}")
```

---

## 5.6 Recuperação: dense, sparse e híbrida

### Dense retrieval (busca semântica)

Usa embeddings. Encontra documentos *semanticamente similares* mesmo com palavras diferentes. "Carro" encontra "automóvel". Ótimo para linguagem natural e variações de vocabulário.

**Limitação:** péssimo para termos técnicos específicos, números, códigos de produto, nomes próprios. "CVE-2024-1234" vai se perder em embedding semântico.

### Sparse retrieval (BM25)

Algoritmo clássico baseado em frequência de termos. Excelente para *correspondência exata de palavras-chave*. Ainda é amplamente usado em mecanismos de busca tradicionais.

**Limitação:** não entende sinônimos ou variações semânticas.

### Híbrida: o melhor dos dois mundos

Em produção, a maioria dos sistemas sérios usa busca híbrida: combina os scores de dense e sparse para obter melhores resultados.

```python
import math


class BM25Simple:
    """Implementação minimalista de BM25 para demonstração."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0

    def fit(self, documents: list[str]) -> None:
        self.docs = [doc.lower().split() for doc in documents]
        N = len(self.docs)
        self.avg_dl = sum(len(d) for d in self.docs) / N if N > 0 else 0

        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        terms = query.lower().split()
        doc = self.docs[doc_idx]
        dl = len(doc)
        score = 0.0

        term_freq: dict[str, int] = {}
        for term in doc:
            term_freq[term] = term_freq.get(term, 0) + 1

        for term in terms:
            if term not in self.idf:
                continue
            tf = term_freq.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf[term] * (numerator / denominator)

        return score

    def search(self, query: str, n: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


def hybrid_search(
    query: str,
    documents: list[str],
    dense_scores: list[float],
    bm25: BM25Simple,
    n_results: int = 5,
    alpha: float = 0.5,
) -> list[dict]:
    """
    Combina scores dense e sparse com normalização min-max.
    alpha=0: só sparse, alpha=1: só dense.
    """
    sparse_results = bm25.search(query, n=len(documents))
    sparse_score_map = {idx: score for idx, score in sparse_results}

    def normalize(scores: list[float]) -> list[float]:
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    dense_norm = normalize(dense_scores)
    all_sparse = [sparse_score_map.get(i, 0.0) for i in range(len(documents))]
    sparse_norm = normalize(all_sparse)

    results = []
    for i, doc in enumerate(documents):
        hybrid_score = alpha * dense_norm[i] + (1 - alpha) * sparse_norm[i]
        results.append({
            "doc_idx": i,
            "content": doc,
            "hybrid_score": hybrid_score,
            "dense_score": dense_norm[i],
            "sparse_score": sparse_norm[i],
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:n_results]
```

### Reranking: refinando os top-K resultados

Após retrieval, um cross-encoder pode re-ranquear os top-K resultados com mais precisão. Cross-encoders são mais lentos mas mais precisos que bi-encoders.

```python
def rerank_with_cross_encoder(
    query: str,
    candidates: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [(query, candidate) for candidate in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]
```

O fluxo completo: retrieval amplo (top-20) → reranking cross-encoder → top-3/5 para o LLM. Isso melhora significativamente a qualidade sem explodir o contexto.

---

## 5.7 Avaliação do pipeline RAG

Você não pode melhorar o que não mede. Avaliação de RAG acontece em duas camadas: qualidade do retrieval e qualidade da resposta.

### Métricas de retrieval

| Métrica | O que mede | Como calcular |
|---------|-----------|---------------|
| **Precision@K** | Dos K chunks retornados, quantos são relevantes? | `relevantes_entre_K / K` |
| **Recall@K** | Dos chunks relevantes existentes, quantos foram recuperados? | `recuperados_relevantes / total_relevantes` |
| **MRR** | Qual a posição média do primeiro resultado relevante? | `1 / posição_do_primeiro_relevante` |

### Ragas: avaliação end-to-end automatizada

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


def evaluate_rag_pipeline(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    questions:     perguntas do usuário
    answers:       respostas geradas pelo LLM
    contexts:      chunks recuperados para cada pergunta (lista de listas)
    ground_truths: respostas corretas de referência
    """
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,        # resposta é suportada pelo contexto?
            answer_relevancy,    # resposta é relevante à pergunta?
            context_precision,   # contexto recuperado é preciso?
            context_recall,      # contexto captura o que é necessário?
        ],
    )

    return result


def evaluate_retrieval(
    retrieval_fn,
    test_cases: list[dict],
    k: int = 5,
) -> dict:
    """
    Harness de avaliação de retrieval sem dependências externas.

    test_cases: lista de {"query": str, "relevant_doc_ids": list[str]}
    retrieval_fn: função que recebe query e retorna lista de {"doc_id": str}
    """
    precisions, recalls, mrrs = [], [], []

    for case in test_cases:
        query = case["query"]
        relevant = set(case["relevant_doc_ids"])
        retrieved = retrieval_fn(query, k=k)
        retrieved_ids = [r["doc_id"] for r in retrieved]

        # Precision@K
        relevant_retrieved = [id for id in retrieved_ids[:k] if id in relevant]
        precision = len(relevant_retrieved) / k
        precisions.append(precision)

        # Recall@K
        recall = len(relevant_retrieved) / len(relevant) if relevant else 0
        recalls.append(recall)

        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

    return {
        f"precision@{k}": sum(precisions) / len(precisions),
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
    }


# Exemplo de conjunto de teste manual
TEST_CASES = [
    {
        "query": "Qual é a política de férias?",
        "relevant_doc_ids": ["rh_politicas_chunk_3", "rh_politicas_chunk_4"],
    },
    {
        "query": "Como solicitar vale-refeição?",
        "relevant_doc_ids": ["rh_beneficios_chunk_1"],
    },
]
```

---

## 5.8 Dados obsoletos e atualização do índice

Documentos mudam. A política de RH atualiza. A API muda de versão. Seu índice precisa refletir isso.

### Estratégias de atualização

**Delete-and-reinsert** é a abordagem mais simples e mais comum:

```python
import time


def update_document_in_index(
    collection,
    embedding_service,
    new_document_path: str,
    source_id: str,
) -> None:
    """Atualiza um documento: remove chunks antigos, indexa novos."""
    from pathlib import Path

    # 1. Remove versão antiga
    delete_by_source(collection, source_id)

    # 2. Extrai e processa nova versão
    path = Path(new_document_path)
    doc = extract_pdf(path) or extract_markdown(path)
    if not doc:
        raise ValueError(f"Não foi possível extrair: {new_document_path}")

    doc.content = clean_text(doc.content)
    chunks_raw = chunk_by_paragraph(doc.content)

    if not chunks_raw:
        raise ValueError(f"Nenhum chunk gerado para: {new_document_path}")

    # 3. Gera embeddings
    embeddings = embedding_service.embed_batch(chunks_raw)

    # 4. Indexa com timestamp de atualização
    metadatas = [
        {
            "source": source_id,
            "updated_at": int(time.time()),
            "chunk_index": i,
        }
        for i in range(len(chunks_raw))
    ]
    index_chunks(collection, chunks_raw, metadatas, embeddings)

    print(f"Documento '{source_id}' atualizado: {len(chunks_raw)} chunks.")
```

### Filtragem por recência

Adicione `updated_at` como metadado e filtre na query:

```python
import time

# Só documentos atualizados nos últimos 90 dias
cutoff = int(time.time()) - (90 * 24 * 60 * 60)
results = search(
    collection,
    query_embedding,
    where={"updated_at": {"$gte": cutoff}},
)
```

### Monitorando a frescura do índice

```python
def check_index_freshness(collection, max_age_days: int = 30) -> dict:
    """Verifica se há documentos com mais de N dias sem atualização."""
    cutoff = int(time.time()) - (max_age_days * 24 * 60 * 60)
    all_docs = collection.get(include=["metadatas"])

    stale_sources = set()
    for meta in all_docs["metadatas"]:
        updated_at = meta.get("updated_at", 0)
        if updated_at < cutoff:
            stale_sources.add(meta.get("source", "desconhecido"))

    return {
        "stale_sources": list(stale_sources),
        "stale_count": len(stale_sources),
        "needs_attention": len(stale_sources) > 0,
    }
```

---

## 5.9 Modos de falha comuns em RAG

Conhecer os modos de falha economiza horas de debug em produção.

| Falha | Sintoma | Causa raiz | Solução |
|-------|---------|-----------|---------|
| **Retrieval failure** | Resposta correta existe nos docs mas não é usada | Chunks errados recuperados | Melhore chunking, revise embeddings, use híbrida |
| **Context irrelevance** | Chunks recuperados mas não úteis | Embeddings ruins para o domínio/idioma | Teste modelos multilíngues específicos |
| **Alucinação com contexto** | Modelo ignora o contexto e inventa | Instrução fraca no prompt ou contexto muito longo | Fortaleça instrução, reduza tamanho do contexto |
| **Artefato de chunking** | Resposta incompleta, frase cortada no meio | Chunk dividido em ponto ruim | Use overlap maior, chunking semântico |
| **Informação contraditória** | Resposta confusa ou inconsistente | Múltiplas versões do mesmo doc no índice | Limpeza do índice, política de update clara |
| **Latência alta** | Usuário espera muito | Embedding na query + retrieval + LLM em sequência | Cache de embeddings de queries frequentes |

### Template de system prompt para RAG com instrução forte

```python
RAG_SYSTEM_PROMPT = """Você é um assistente que responde perguntas com base nos documentos fornecidos.

REGRAS ESTRITAS:
1. Responda SOMENTE com base nas informações presentes nos documentos abaixo.
2. Se a informação não estiver nos documentos, diga explicitamente: "Não encontrei essa informação nos documentos disponíveis."
3. Nunca invente informações ou complete com conhecimento externo.
4. Quando citar uma informação, indique de qual documento ela veio.

DOCUMENTOS:
{context}
"""

def build_rag_prompt(retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"].get("source", f"Documento {i}")
        context_parts.append(f"[{i}] Fonte: {source}\n{chunk['content']}")

    return RAG_SYSTEM_PROMPT.format(context="\n\n---\n\n".join(context_parts))
```

---

## 📌 Resumo da Parte 05

| Conceito | Definição |
|----------|-----------|
| **RAG** | Recuperação de documentos relevantes para aumentar o contexto do LLM com conhecimento externo |
| **Ingestion** | Extração, limpeza e preparação de documentos para indexação |
| **Chunking** | Divisão de documentos em unidades recuperáveis; tamanho depende do domínio |
| **Embedding** | Representação vetorial do texto que permite busca por similaridade semântica |
| **Dense retrieval** | Busca por similaridade de embedding; bom para variações semânticas |
| **Sparse retrieval (BM25)** | Busca por palavras-chave; bom para termos técnicos exatos |
| **Busca híbrida** | Combinação de dense e sparse; melhor resultado na maioria dos casos |
| **Reranking** | Refinamento dos top-K resultados com cross-encoder; mais preciso, mais lento |
| **Precision@K** | Proporção de resultados relevantes entre os top-K recuperados |
| **Recall@K** | Proporção de documentos relevantes que foram recuperados nos top-K |
| **Filtragem de metadados** | Restrição da busca por atributos como categoria, data, fonte |
| **Ragas** | Biblioteca para avaliação automatizada de pipelines RAG |

## 🔗 Referências

- [MTEB Leaderboard — benchmarks de embedding](https://huggingface.co/spaces/mteb/leaderboard)
- [Ragas — avaliação de pipelines RAG](https://github.com/explodinggradients/ragas)
- [ChromaDB documentação](https://docs.trychroma.com/)
- [Qdrant documentação](https://qdrant.tech/documentation/)
- [BAAI/bge-m3 — modelo multilíngue](https://huggingface.co/BAAI/bge-m3)
- [Pinecone — guia de chunking](https://www.pinecone.io/learn/chunking-strategies/)
- [Advanced RAG — survey paper](https://arxiv.org/abs/2312.10997)
- [pgvector — extensão vetorial para PostgreSQL](https://github.com/pgvector/pgvector)

---

⬅️ **Anterior:** [Parte 04](./parte-04-engenharia-de-contexto-2.md) | ➡️ **Próximo:** [Parte 06](./parte-06-agentes-no-sistema.md)  
🏠 **Início:** [README](../README.md)
