# Prática 05 — RAG Simples: Seu Primeiro Sistema de Perguntas e Respostas

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 07](../conteudo/parte-07-rag.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Implementar um pipeline RAG completo do zero
- Indexar documentos reais em um banco vetorial
- Conectar a recuperação de documentos com geração de resposta
- Avaliar a qualidade das respostas com e sem RAG

---

## 🔧 Setup

```bash
pip install openai chromadb python-dotenv sentence-transformers
```

---

## 📝 Exercício 1 — RAG vs Sem RAG

Crie `pratica05/ex01_rag_vs_sem_rag.py` para ver a diferença concreta:

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

---

## 📝 Exercício 2 — RAG com Pontuação de Confiança

Crie `pratica05/ex02_rag_com_confianca.py`:

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# (reutiliza o setup do ex01 — adapte conforme necessário)

def rag_com_confianca(pergunta: str, collection, threshold: float = 0.5) -> dict:
    """RAG que indica quando não encontrou informação suficiente."""
    
    results = collection.query(query_texts=[pergunta], n_results=3)
    
    docs = results["documents"][0]
    dists = results["distances"][0]
    
    # Distância 0 = idêntico, 1 = completamente diferente (espaço cosseno)
    scores = [1 - d for d in dists]  # converter para similaridade
    
    print(f"\n  Top chunks recuperados:")
    for doc, score in zip(docs, scores):
        print(f"    [{score:.3f}] {doc[:60]}...")
    
    # Se o melhor match é muito baixo, não responder com base nos docs
    melhor_score = max(scores)
    if melhor_score < threshold:
        return {
            "resposta": "Não encontrei informações suficientes para responder com precisão.",
            "confianca": melhor_score,
            "respondeu": False
        }
    
    # Filtrar apenas docs com boa relevância
    docs_relevantes = [d for d, s in zip(docs, scores) if s >= threshold * 0.8]
    contexto = "\n".join(docs_relevantes)
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Com base APENAS no contexto abaixo, responda a pergunta.
Retorne JSON com: {{"resposta": "...", "base_no_contexto": true/false}}

CONTEXTO: {contexto}
PERGUNTA: {pergunta}"""
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(resp.choices[0].message.content)
    result["confianca"] = melhor_score
    result["respondeu"] = True
    return result
```

---

## 📝 Exercício 3 — RAG para Documentação de Código

Crie `pratica05/ex03_rag_documentacao.py`:

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Documentação de uma API fictícia
DOCUMENTACAO = [
    """
    GET /users
    Retorna lista de usuários. 
    Query params: page (int), limit (int, max 100), active (bool).
    Response: {"users": [...], "total": int, "page": int}
    Auth: Bearer token obrigatório.
    """,
    """
    POST /users
    Cria um novo usuário.
    Body: {"name": str (required), "email": str (required), "role": "admin"|"user" (default: user)}
    Response: {"id": str, "name": str, "email": str, "created_at": timestamp}
    Auth: Bearer token com permissão admin.
    """,
    """
    GET /users/{id}
    Retorna um usuário específico pelo ID.
    Path: id (UUID obrigatório).
    Response: objeto User completo ou 404 se não encontrado.
    Auth: Bearer token. Usuário comum só pode ver a si mesmo.
    """,
    """
    PUT /users/{id}
    Atualiza dados de um usuário.
    Body: {"name": str (opcional), "email": str (opcional), "role": str (apenas admin)}.
    Campos não enviados não são alterados.
    Auth: Bearer token. Admin pode atualizar qualquer usuário.
    """,
    """
    DELETE /users/{id}
    Remove um usuário. Operação irreversível.
    Retorna 204 No Content em caso de sucesso.
    Auth: Bearer token com permissão admin obrigatória.
    Não é possível deletar a si mesmo.
    """,
    """
    POST /auth/login
    Autentica um usuário e retorna token JWT.
    Body: {"email": str, "password": str}
    Response: {"token": str, "expires_in": int (segundos), "refresh_token": str}
    Sem autenticação prévia.
    """,
    """
    Erros comuns da API:
    400 Bad Request: dados de entrada inválidos
    401 Unauthorized: token ausente ou inválido
    403 Forbidden: sem permissão para a operação
    404 Not Found: recurso não encontrado
    429 Too Many Requests: rate limit excedido (100 req/min)
    500 Internal Server Error: erro no servidor
    """,
]

chroma = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
docs_collection = chroma.create_collection("api_docs", embedding_function=ef)
docs_collection.add(
    documents=DOCUMENTACAO,
    ids=[f"doc{i}" for i in range(len(DOCUMENTACAO))]
)

def consultar_docs(pergunta: str) -> str:
    """Consulta a documentação da API."""
    results = docs_collection.query(query_texts=[pergunta], n_results=3)
    contexto = "\n\n".join(results["documents"][0])
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Você é um assistente técnico especializado nesta API.
Responda com base na documentação fornecida. Seja preciso e inclua exemplos de código quando útil.
Se a informação não estiver na documentação, diga claramente."""
            },
            {
                "role": "user",
                "content": f"Documentação relevante:\n{contexto}\n\nPergunta: {pergunta}"
            }
        ]
    )
    return resp.choices[0].message.content

# Testar
perguntas_dev = [
    "Como faço para criar um novo usuário?",
    "Quais permissões são necessárias para deletar um usuário?",
    "O que acontece se eu fazer mais de 100 requisições por minuto?",
    "Como obtenho um token de autenticação?",
    "Posso filtrar usuários por status ativo?",
]

for p in perguntas_dev:
    print(f"\n❓ {p}")
    print(consultar_docs(p))
    print()
```

---

## 📝 Exercício 4 — Chatbot RAG Completo (Projeto Principal)

Crie `pratica05/chatbot_rag.py`:

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

## 🏆 Desafios Opcionais

1. **Indexar PDFs**: Use `pypdf` para extrair texto de PDFs e indexá-los
2. **Chunking melhorado**: Implemente chunking por parágrafo em vez de tamanho fixo
3. **Citar fontes**: Modifique o chatbot para sempre citar qual documento embasou a resposta
4. **Modo de avaliação**: Implemente um modo onde você testa perguntas com respostas esperadas

---

## ✅ Checklist de Entrega

- [ ] Ex01: demonstração RAG vs sem RAG funcionando
- [ ] Ex02: RAG com pontuação de confiança
- [ ] Ex03: RAG para documentação de API
- [ ] Chatbot RAG com histórico funcionando
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 04](./pratica-04-banco-vetorial.md) | ➡️ **Próxima:** [Prática 06 — RAG Avançado](./pratica-06-rag-avancado.md)
