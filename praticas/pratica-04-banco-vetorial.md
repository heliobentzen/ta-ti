# Prática 04 — Banco de Dados Vetorial com ChromaDB

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 06](../conteudo/parte-06-bancos-vetoriais.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Criar e gerenciar coleções no ChromaDB
- Indexar, atualizar e deletar documentos
- Realizar buscas com filtros de metadados
- Construir uma mini base de conhecimento persistente

---

## 🔧 Setup

```bash
pip install chromadb openai python-dotenv
```

---

## 📝 Exercício 1 — CRUD com ChromaDB

Crie `pratica04/ex01_crud_chroma.py`:

```python
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────

# Banco persistente em disco
client = chromadb.PersistentClient(path="./pratica04/chroma_db")

# Função de embedding (usa modelo local gratuito)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Criar (ou acessar se já existir) uma coleção
collection = client.get_or_create_collection(
    name="artigos_tech",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

print(f"📦 Coleção: {collection.name} | Documentos: {collection.count()}")

# ─────────────────────────────────────────
# CREATE — Adicionar documentos
# ─────────────────────────────────────────

artigos = [
    {
        "id": "art001",
        "content": "Python é uma linguagem de programação versátil usada em IA, web e automação.",
        "metadata": {"categoria": "linguagem", "nivel": "iniciante", "ano": 2024}
    },
    {
        "id": "art002", 
        "content": "Machine Learning permite que computadores aprendam padrões sem serem explicitamente programados.",
        "metadata": {"categoria": "ia", "nivel": "intermediario", "ano": 2024}
    },
    {
        "id": "art003",
        "content": "RAG combina recuperação de documentos com geração de texto por LLMs para respostas precisas.",
        "metadata": {"categoria": "ia", "nivel": "avancado", "ano": 2024}
    },
    {
        "id": "art004",
        "content": "Docker permite empacotar aplicações em containers para portabilidade e escalabilidade.",
        "metadata": {"categoria": "devops", "nivel": "intermediario", "ano": 2023}
    },
    {
        "id": "art005",
        "content": "APIs REST usam verbos HTTP (GET, POST, PUT, DELETE) para operações em recursos.",
        "metadata": {"categoria": "web", "nivel": "iniciante", "ano": 2023}
    },
]

if collection.count() == 0:
    collection.add(
        documents=[a["content"] for a in artigos],
        ids=[a["id"] for a in artigos],
        metadatas=[a["metadata"] for a in artigos]
    )
    print(f"✅ {len(artigos)} artigos adicionados!")
else:
    print(f"ℹ️  Coleção já tem {collection.count()} documentos")

# ─────────────────────────────────────────
# READ — Consultar documentos
# ─────────────────────────────────────────

print("\n📖 Busca semântica: 'como usar inteligência artificial'")
results = collection.query(
    query_texts=["como usar inteligência artificial"],
    n_results=3
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"  [{dist:.3f}] [{meta['categoria']}] {doc[:80]}...")

# ─────────────────────────────────────────
# UPDATE — Atualizar um documento
# ─────────────────────────────────────────

collection.update(
    ids=["art001"],
    documents=["Python é uma linguagem de programação de alto nível, versátil e com sintaxe clara, amplamente usada em IA, ciência de dados, web e automação."],
    metadatas=[{"categoria": "linguagem", "nivel": "iniciante", "ano": 2024, "atualizado": True}]
)
print("\n✏️  Documento art001 atualizado!")

# ─────────────────────────────────────────
# DELETE — Remover um documento
# ─────────────────────────────────────────

# collection.delete(ids=["art005"])
# print("🗑️  Documento art005 removido!")

print(f"\n📊 Total de documentos: {collection.count()}")
```

---

## 📝 Exercício 2 — Busca com Filtros de Metadados

Crie `pratica04/ex02_filtros.py`:

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./pratica04/chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection("artigos_tech", embedding_function=ef)

consulta = "como aprender programação"

print(f"🔍 Consulta: '{consulta}'\n")

# Filtro 1: apenas nível iniciante
print("=== FILTRO: apenas artigos para iniciantes ===")
results = collection.query(
    query_texts=[consulta],
    n_results=5,
    where={"nivel": "iniciante"}
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['nivel']}] {doc[:70]}...")

# Filtro 2: IA ou linguagem
print("\n=== FILTRO: categoria IA ou linguagem ===")
results = collection.query(
    query_texts=[consulta],
    n_results=5,
    where={"categoria": {"$in": ["ia", "linguagem"]}}
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['categoria']}] {doc[:70]}...")

# Filtro 3: ano >= 2024
print("\n=== FILTRO: publicados em 2024 ===")
results = collection.query(
    query_texts=[consulta],
    n_results=5,
    where={"ano": {"$gte": 2024}}
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['ano']}] {doc[:70]}...")

# Filtro 4: não é devops E não é avançado
print("\n=== FILTRO: não é devops E não é avançado ===")
results = collection.query(
    query_texts=[consulta],
    n_results=5,
    where={
        "$and": [
            {"categoria": {"$ne": "devops"}},
            {"nivel": {"$ne": "avancado"}}
        ]
    }
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['categoria']}|{meta['nivel']}] {doc[:70]}...")
```

---

## 📝 Exercício 3 — Pipeline de Indexação de Arquivos

Crie `pratica04/ex03_indexar_arquivos.py`:

```python
import chromadb
from chromadb.utils import embedding_functions
import os
import hashlib

client = chromadb.PersistentClient(path="./pratica04/chroma_docs")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection("documentos", embedding_function=ef)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Divide texto em chunks com sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk) > 50:
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def indexar_texto(texto: str, nome_fonte: str, metadados: dict = None):
    """Indexa um texto dividindo em chunks."""
    chunks = chunk_text(texto)
    
    if not chunks:
        return 0
    
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        ids.append(f"{nome_fonte}_chunk{i}_{chunk_hash}")
    
    base_meta = {"fonte": nome_fonte, "total_chunks": len(chunks)}
    if metadados:
        base_meta.update(metadados)
    
    metas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]
    
    collection.upsert(documents=chunks, ids=ids, metadatas=metas)
    return len(chunks)

# Indexar textos de exemplo (simula indexação de arquivos)
textos = [
    {
        "nome": "manual_python",
        "meta": {"tipo": "manual", "linguagem": "Python"},
        "conteudo": """
Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.
Foi criada por Guido van Rossum e lançada em 1991.

Principais características:
- Sintaxe clara e legível
- Tipagem dinâmica
- Gerenciamento automático de memória
- Grande ecossistema de bibliotecas

Python é amplamente usado em ciência de dados, machine learning, desenvolvimento web,
automação e scripting. Bibliotecas como NumPy, Pandas, TensorFlow e Django tornaram
Python uma das linguagens mais populares do mundo.

Para instalar Python, acesse python.org e baixe a versão mais recente.
Recomenda-se usar ambientes virtuais para gerenciar dependências.
        """
    },
    {
        "nome": "intro_ia",
        "meta": {"tipo": "artigo", "area": "ia"},
        "conteudo": """
Inteligência Artificial (IA) é a simulação de processos de inteligência humana por sistemas computacionais.

Subcampos principais:
1. Machine Learning: sistemas que aprendem com dados
2. Deep Learning: redes neurais profundas
3. NLP: processamento de linguagem natural
4. Computer Vision: análise de imagens

Os LLMs (Large Language Models) revolucionaram a IA generativa a partir de 2022.
Modelos como GPT-4, Claude e Gemini conseguem gerar texto, código e imagens de alta qualidade.

O paradigma RAG (Retrieval Augmented Generation) permite conectar LLMs a bases de conhecimento
personalizadas, superando a limitação do corte de conhecimento dos modelos.
        """
    }
]

for t in textos:
    n = indexar_texto(t["conteudo"], t["nome"], t["meta"])
    print(f"✅ {t['nome']}: {n} chunks indexados")

print(f"\n📦 Total na coleção: {collection.count()} chunks")

# Testar busca
print("\n🔍 Busca: 'como instalar Python'")
results = collection.query(query_texts=["como instalar Python"], n_results=3)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['fonte']}] {doc[:100]}...")
```

---

## 📝 Exercício 4 — Gerenciador de Notas Inteligente (Projeto Principal)

Crie `pratica04/notas_inteligentes.py`:

```python
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import hashlib
import json

class GerenciadorNotas:
    """Sistema de notas com busca semântica."""
    
    def __init__(self, db_path: str = "./pratica04/notas"):
        self.client = chromadb.PersistentClient(path=db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            "notas",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
    
    def adicionar(self, titulo: str, conteudo: str, tags: list[str] = None) -> str:
        """Adiciona uma nota."""
        nota_id = f"nota_{hashlib.md5(conteudo.encode()).hexdigest()[:10]}"
        metadata = {
            "titulo": titulo,
            "tags": ",".join(tags or []),
            "criado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "palavras": len(conteudo.split())
        }
        
        self.collection.upsert(
            documents=[f"{titulo}\n\n{conteudo}"],
            ids=[nota_id],
            metadatas=[metadata]
        )
        print(f"✅ Nota adicionada: '{titulo}' (ID: {nota_id})")
        return nota_id
    
    def buscar(self, consulta: str, top_k: int = 3, tag_filter: str = None) -> list:
        """Busca notas por conteúdo semântico."""
        where = None
        if tag_filter:
            where = {"tags": {"$contains": tag_filter}}
        
        results = self.collection.query(
            query_texts=[consulta],
            n_results=min(top_k, self.collection.count()),
            where=where
        )
        
        return [
            {
                "titulo": meta["titulo"],
                "conteudo": doc[:200],
                "tags": meta["tags"],
                "criado_em": meta["criado_em"],
                "score": round(1 - dist, 3)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def listar(self) -> list:
        """Lista todas as notas."""
        if self.collection.count() == 0:
            return []
        all_notes = self.collection.get(include=["metadatas"])
        return all_notes["metadatas"]
    
    def total(self) -> int:
        return self.collection.count()


# Demo
gn = GerenciadorNotas()

if gn.total() == 0:
    # Adicionar notas de exemplo
    gn.adicionar("Aula 1: LLMs", 
                 "LLMs são modelos de linguagem treinados em bilhões de tokens. Usam arquitetura Transformer.",
                 tags=["ia", "llm", "aula"])
    
    gn.adicionar("RAG - Conceitos",
                 "RAG significa Retrieval Augmented Generation. Combina busca vetorial com LLMs.",
                 tags=["ia", "rag", "aula"])
    
    gn.adicionar("Receita de bolo de chocolate",
                 "Ingredientes: farinha, açúcar, ovos, manteiga, chocolate em pó. Misture e leve ao forno.",
                 tags=["culinaria", "receita"])
    
    gn.adicionar("Ideas for project",
                 "Build a RAG chatbot for university documents. Use ChromaDB + GPT-4o-mini.",
                 tags=["projeto", "ia"])

print(f"\n📝 Total de notas: {gn.total()}\n")

# Buscar
consultas = [
    "como funcionam os modelos de linguagem",
    "sistema de recuperação de documentos",
    "culinária e comida",
]

for c in consultas:
    print(f"\n🔍 '{c}':")
    notas = gn.buscar(c, top_k=2)
    for n in notas:
        print(f"  [{n['score']}] {n['titulo']} ({n['tags']})")

# Interface interativa simples
print("\n" + "="*50)
print("🔍 BUSCA INTERATIVA (Ctrl+C para sair)")
while True:
    try:
        query = input("\nBuscar: ").strip()
        if query:
            resultados = gn.buscar(query)
            for r in resultados:
                print(f"  [{r['score']}] {r['titulo']}")
                print(f"  {r['conteudo'][:100]}...")
    except KeyboardInterrupt:
        print("\nEncerrando...")
        break
```

---

## 🏆 Desafios Opcionais

1. **Múltiplas coleções**: Crie um sistema com coleções separadas por categoria
2. **Backup e restore**: Exporte e importe documentos em JSON
3. **Estatísticas**: Calcule estatísticas da coleção (documentos por categoria, palavra mais comum, etc.)
4. **Interface web**: Use Streamlit para criar uma interface visual para o gerenciador de notas

---

## ✅ Checklist de Entrega

- [ ] Ex01: CRUD completo funcionando
- [ ] Ex02: filtros de metadados funcionando
- [ ] Ex03: indexação de textos em chunks
- [ ] Gerenciador de notas funcionando
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 03](./pratica-03-embeddings.md) | ➡️ **Próxima:** [Prática 05 — RAG Simples](./pratica-05-rag-simples.md)
