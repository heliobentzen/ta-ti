# Prática 05 — Conhecimento Externo e RAG

> **Carga horária estimada:** 3 horas  
> **Conteúdo relacionado:** [Parte 05](../conteudo/parte-05-conhecimento-externo-rag.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Gerar embeddings e calcular similaridade semântica entre textos
- Construir um buscador semântico com a API da OpenAI
- Criar e gerenciar coleções no ChromaDB com indexação em chunks
- Implementar um pipeline RAG completo com histórico de conversa
- Aplicar reranking com cross-encoder para melhorar a precisão da recuperação

---

## 🔧 Setup

```bash
pip install openai chromadb python-dotenv sentence-transformers numpy scikit-learn rank-bm25
```

Configure o arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sk-...
```

---

## 📝 Exercício 1 — Embeddings e Busca Semântica

Neste exercício, você vai gerar embeddings com a API da OpenAI, calcular similaridade entre textos e construir um buscador semântico para uma base de perguntas frequentes.

### Parte A — Gerando e Comparando Embeddings

Crie `pratica03/ex01a_gerando_embeddings.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Gera embedding para um texto usando OpenAI."""
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Calcula similaridade de cosseno entre dois vetores."""
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Pares de textos para comparar
pares = [
    ("gato", "felino"),
    ("gato", "cachorro"),
    ("gato", "automóvel"),
    ("Python é uma linguagem de programação", "Java é uma linguagem de programação"),
    ("Python é uma linguagem de programação", "O Brasil é um país sul-americano"),
    ("Inteligência Artificial", "IA"),
    ("Como faço login?", "Esqueci minha senha"),
    ("Como faço login?", "Qual é o preço do plano?"),
]

print("Calculando similaridades...\n")
print(f"{'Texto A':<35} {'Texto B':<35} {'Similaridade'}")
print("-" * 85)

for texto_a, texto_b in pares:
    emb_a = get_embedding(texto_a)
    emb_b = get_embedding(texto_b)
    sim = cosine_similarity(emb_a, emb_b)
    
    # Interpretação
    if sim > 0.85:
        status = "🟢 Muito similar"
    elif sim > 0.70:
        status = "🟡 Similar"
    elif sim > 0.50:
        status = "🟠 Pouco similar"
    else:
        status = "🔴 Diferente"
    
    print(f"{texto_a[:33]:<35} {texto_b[:33]:<35} {sim:.3f} {status}")
```

### Parte B — Buscador Semântico

Crie `pratica03/ex01b_busca_semantica.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()
client = OpenAI()

# Base de conhecimento (perguntas frequentes)
FAQ = [
    {
        "id": 1,
        "pergunta": "Como faço para redefinir minha senha?",
        "resposta": "Acesse a tela de login e clique em 'Esqueci minha senha'. Você receberá um email com instruções."
    },
    {
        "id": 2,
        "pergunta": "Quais são os métodos de pagamento aceitos?",
        "resposta": "Aceitamos cartão de crédito (Visa, Master, Amex), PIX e boleto bancário."
    },
    {
        "id": 3,
        "pergunta": "Como cancelo minha assinatura?",
        "resposta": "Você pode cancelar em Configurações > Assinatura > Cancelar plano. O acesso continua até o fim do período."
    },
    {
        "id": 4,
        "pergunta": "O produto funciona offline?",
        "resposta": "Sim! Você pode usar o app offline. Os dados sincronizam quando a conexão for restaurada."
    },
    {
        "id": 5,
        "pergunta": "Como exporto meus dados?",
        "resposta": "Em Configurações > Dados > Exportar, você pode baixar todos os seus dados em formato CSV ou JSON."
    },
    {
        "id": 6,
        "pergunta": "Tem plano gratuito?",
        "resposta": "Sim! Oferecemos um plano gratuito com até 100 itens. Planos pagos começam em R$29/mês."
    },
    {
        "id": 7,
        "pergunta": "Como entro em contato com o suporte?",
        "resposta": "Pelo chat dentro do app, email suporte@empresa.com ou WhatsApp (81) 99999-0000."
    },
]

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings em lote."""
    response = client.embeddings.create(
        input=[t.replace("\n", " ") for t in texts],
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Indexar FAQ
print("Indexando base de conhecimento...")
perguntas = [item["pergunta"] for item in FAQ]
embeddings_faq = embed_texts(perguntas)

def buscar(consulta: str, top_k: int = 3) -> list[dict]:
    """Busca as respostas mais relevantes para uma consulta."""
    embedding_consulta = embed_texts([consulta])[0]
    
    scores = [
        (cosine_similarity(embedding_consulta, emb), faq)
        for emb, faq in zip(embeddings_faq, FAQ)
    ]
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    return [
        {"score": round(score, 3), **faq}
        for score, faq in scores[:top_k]
    ]

# Testar buscas
consultas = [
    "não consigo entrar na minha conta",
    "posso pagar com PIX?",
    "quero parar de usar o serviço",
    "tem versão sem internet?",
    "quanto custa?",
]

for consulta in consultas:
    print(f"\n🔍 Consulta: '{consulta}'")
    resultados = buscar(consulta, top_k=2)
    for r in resultados:
        print(f"  [{r['score']}] {r['pergunta']}")
        print(f"  → {r['resposta']}")
```

---

## 📝 Exercício 2 — Banco Vetorial com ChromaDB

Agora que você sabe gerar embeddings e calcular similaridade, vamos usar o ChromaDB como banco vetorial para indexar, buscar e gerenciar documentos de forma persistente.

### Parte A — CRUD com ChromaDB

Crie `pratica03/ex02a_crud_chroma.py`:

```python
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────

# Banco persistente em disco
client = chromadb.PersistentClient(path="./pratica03/chroma_db")

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

### Parte B — Pipeline de Indexação de Arquivos

Crie `pratica03/ex02b_indexar_arquivos.py`:

```python
import chromadb
from chromadb.utils import embedding_functions
import os
import hashlib

client = chromadb.PersistentClient(path="./pratica03/chroma_docs")
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

## 📝 Exercício 3 — RAG Pipeline Completo

Com embeddings e banco vetorial dominados, vamos montar um pipeline RAG completo: recuperação de documentos relevantes + geração de resposta por LLM.

### Parte A — RAG vs Sem RAG

Crie `pratica03/ex03a_rag_vs_sem_rag.py` para ver a diferença concreta:

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Base de conhecimento privada (informações que o LLM não sabe)
DOCUMENTOS_EMPRESA = [
    "Horário de funcionamento: Segunda a sexta das 8h às 18h. Sábados das 9h às 13h.",
    "Política de devolução: Produtos podem ser devolvidos em até 30 dias com nota fiscal.",
    "Planos disponíveis: Basic (R$29/mês), Pro (R$79/mês), Enterprise (sob consulta).",
    "O plano Pro inclui: 50GB de armazenamento, suporte prioritário e API com 10k req/mês.",
    "Suporte técnico: chat online 24/7, email suporte@empresa.com, telefone (81) 3000-0000.",
    "CEO: Ana Ferreira, fundada em 2020 em Recife-PE.",
    "Tecnologia: stack Python/FastAPI no backend, React no frontend, AWS na nuvem.",
    "Parceiros oficiais: Google Cloud, AWS, Microsoft Azure.",
]

# Configurar ChromaDB com embeddings locais (sem custo)
chroma = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma.create_collection("empresa", embedding_function=ef)

# Indexar documentos
collection.add(
    documents=DOCUMENTOS_EMPRESA,
    ids=[f"doc{i}" for i in range(len(DOCUMENTOS_EMPRESA))]
)

def responder_sem_rag(pergunta: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente de atendimento."},
            {"role": "user", "content": pergunta}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

def responder_com_rag(pergunta: str) -> str:
    # Recuperar contexto relevante
    results = collection.query(query_texts=[pergunta], n_results=3)
    contexto = "\n".join(results["documents"][0])
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Você é um assistente de atendimento.
Use SOMENTE as informações abaixo para responder.
Se não souber, diga que não encontrou a informação.

INFORMAÇÕES DA EMPRESA:
{contexto}"""
            },
            {"role": "user", "content": pergunta}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

# Comparar respostas
perguntas = [
    "Qual é o horário de atendimento?",
    "Quais são os planos disponíveis?",
    "Como entro em contato com o suporte?",
    "Quem fundou a empresa?",
]

for pergunta in perguntas:
    print(f"\n{'='*60}")
    print(f"❓ {pergunta}")
    print(f"\n❌ SEM RAG:")
    print(responder_sem_rag(pergunta))
    print(f"\n✅ COM RAG:")
    print(responder_com_rag(pergunta))
```

### Parte B — Chatbot RAG com Histórico

Crie `pratica03/ex03b_chatbot_rag.py`:

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

# ──────────────────────────────────────────
# 1. CONFIGURAR BASE DE CONHECIMENTO
# ──────────────────────────────────────────

KNOWLEDGE_BASE = [
    # Adicione seus documentos aqui
    "O IFPE (Instituto Federal de Pernambuco) é uma instituição de ensino público federal.",
    "A disciplina de Tópicos Avançados em TI tem carga horária de 26 horas.",
    "O professor Hélio Bentzen ministra a disciplina de IA Generativa.",
    "RAG significa Retrieval Augmented Generation e combina busca com geração.",
    "LLMs são modelos treinados em bilhões de tokens para prever o próximo token.",
    "Embeddings são representações vetoriais de texto com significado semântico.",
    "ChromaDB é um banco de dados vetorial open-source usado para RAG.",
    "A temperatura do modelo controla a criatividade: 0 = determinístico, 1 = criativo.",
    "Agentes de IA usam LLMs para raciocinar e executar ações com ferramentas.",
    "Prompt engineering é a arte de estruturar inputs para obter melhores outputs.",
]

# Setup ChromaDB
chroma = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
kb = chroma.create_collection("chatbot_kb", embedding_function=ef)
kb.add(documents=KNOWLEDGE_BASE, ids=[f"kb{i}" for i in range(len(KNOWLEDGE_BASE))])

print(f"✅ Base de conhecimento carregada: {kb.count()} documentos")

# ──────────────────────────────────────────
# 2. CHATBOT COM RAG E HISTÓRICO
# ──────────────────────────────────────────

class ChatbotRAG:
    def __init__(self, kb_collection, system: str = None):
        self.client = OpenAI()
        self.kb = kb_collection
        self.history = []
        self.system = system or "Você é um assistente útil. Responda com base no contexto fornecido."
    
    def _buscar_contexto(self, query: str, n: int = 3) -> str:
        results = self.kb.query(query_texts=[query], n_results=n)
        docs = results["documents"][0]
        return "\n• " + "\n• ".join(docs) if docs else ""
    
    def chat(self, mensagem: str) -> str:
        # Buscar contexto relevante
        contexto = self._buscar_contexto(mensagem)
        
        # System prompt com contexto
        sys_content = self.system
        if contexto:
            sys_content += f"\n\nCONHECIMENTO DISPONÍVEL:{contexto}"
        
        # Montar mensagens
        messages = [{"role": "system", "content": sys_content}]
        messages += self.history[-6:]  # últimas 3 trocas
        messages += [{"role": "user", "content": mensagem}]
        
        # Gerar resposta
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        resposta = resp.choices[0].message.content
        
        # Atualizar histórico
        self.history.append({"role": "user", "content": mensagem})
        self.history.append({"role": "assistant", "content": resposta})
        
        return resposta

# ──────────────────────────────────────────
# 3. INTERFACE
# ──────────────────────────────────────────

bot = ChatbotRAG(
    kb_collection=kb,
    system="Você é um assistente da disciplina de IA Generativa do IFPE. Seja didático e objetivo."
)

print("\n🤖 Chatbot RAG - Disciplina IA Generativa IFPE")
print("Digite 'sair' para encerrar\n")

while True:
    try:
        user_input = input("Você: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "sair":
            break
        
        resposta = bot.chat(user_input)
        print(f"\n🤖 Bot: {resposta}\n")
    
    except KeyboardInterrupt:
        break

print("Até logo!")
```

---

## 📝 Exercício 4 — RAG Avançado com Reranking (Projeto Principal)

Neste projeto, você vai melhorar a qualidade da recuperação usando um **cross-encoder** para reranking. Cross-encoders são mais precisos que bi-encoders (usados na busca vetorial) porque avaliam query e documento juntos, em vez de separadamente.

Crie `pratica03/ex04_reranking.py`:

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

## 🏆 Desafios Opcionais

1. **Visualização de Embeddings**: Use `matplotlib` para criar um mapa de calor da matriz de similaridade entre textos
2. **Filtros de Metadados**: Adicione filtros por categoria e data nas buscas do ChromaDB (ex: `where={"categoria": "ia"}`)
3. **RAG com Citações**: Modifique o chatbot RAG para sempre citar qual documento embasou cada afirmação
4. **Hybrid Search**: Combine busca BM25 (léxica) com vetorial usando a biblioteca `rank-bm25` para resultados mais robustos
5. **Indexar PDFs**: Use `pypdf` para extrair texto de PDFs reais e indexá-los no ChromaDB

---

## ✅ Checklist de Entrega

- [ ] Ex01: embeddings gerados e buscador semântico de FAQ funcionando
- [ ] Ex02: CRUD no ChromaDB e pipeline de indexação em chunks
- [ ] Ex03: comparação RAG vs sem RAG e chatbot RAG com histórico
- [ ] Ex04: reranking com cross-encoder comparado com busca simples
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 04](./pratica-04-engenharia-de-contexto-2.md) | ➡️ **Próxima:** [Prática 06 — Agentes no Sistema](./pratica-06-agentes.md)
