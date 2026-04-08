# Prática 03 — Gerando e Trabalhando com Embeddings

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 05](../conteudo/parte-05-embeddings.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Gerar embeddings usando a API da OpenAI e modelos locais
- Calcular similaridade entre textos
- Construir um buscador semântico simples
- Detectar duplicatas e agrupar textos por semelhança

---

## 🔧 Setup

```bash
pip install openai python-dotenv sentence-transformers numpy scikit-learn
```

---

## 📝 Exercício 1 — Gerando e Visualizando Embeddings

Crie `pratica03/ex01_gerando_embeddings.py`:

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

---

## 📝 Exercício 2 — Embeddings Locais com Sentence-Transformers

Crie `pratica03/ex02_embeddings_locais.py`:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Modelo leve e eficiente (gratuito, sem API)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Frases em português
frases = [
    "Inteligência artificial está transformando o mercado de trabalho",
    "A IA está mudando como as empresas contratam funcionários",
    "Machine learning é um subcampo da inteligência artificial",
    "Redes neurais aprendem padrões a partir de dados",
    "O futebol é o esporte mais popular do Brasil",
    "A seleção brasileira ganhou 5 copas do mundo",
    "Python é excelente para ciência de dados",
    "R é uma linguagem muito usada em estatística",
]

# Gerar embeddings
print("Gerando embeddings localmente (sem API)...")
embeddings = model.encode(frases, show_progress_bar=True)
print(f"Shape dos embeddings: {embeddings.shape}")  # (8, 384)

# Calcular matriz de similaridade
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(embeddings)

# Encontrar pares mais similares
print("\n📊 TOP 5 PARES MAIS SIMILARES:")
pares_sim = []
for i in range(len(frases)):
    for j in range(i+1, len(frases)):
        pares_sim.append((sim_matrix[i][j], i, j))

pares_sim.sort(reverse=True)
for sim, i, j in pares_sim[:5]:
    print(f"\nSimilaridade: {sim:.3f}")
    print(f"  A: {frases[i]}")
    print(f"  B: {frases[j]}")
```

---

## 📝 Exercício 3 — Buscador Semântico

Crie `pratica03/ex03_busca_semantica.py`:

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

## 📝 Exercício 4 — Detector de Duplicatas (Projeto Principal)

Crie `pratica03/detector_duplicatas.py`:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Simula base de dados de produtos com possíveis duplicatas
produtos = [
    {"id": 1, "nome": "Notebook Dell Inspiron 15 8GB RAM"},
    {"id": 2, "nome": "Dell Inspiron 15 - 8GB memória RAM"},
    {"id": 3, "nome": "Fone de Ouvido Bluetooth JBL Tune 510BT"},
    {"id": 4, "nome": "JBL Tune 510 BT Fone Bluetooth"},
    {"id": 5, "nome": "Mouse Logitech MX Master 3 sem fio"},
    {"id": 6, "nome": "Cadeira Gamer RGB com apoio lombar"},
    {"id": 7, "nome": "Logitech MX Master 3 Mouse Wireless"},
    {"id": 8, "nome": "Monitor LG 27' 4K UHD IPS"},
    {"id": 9, "nome": "Monitor 27 polegadas LG 4K Ultra HD"},
    {"id": 10, "nome": "Teclado Mecânico Redragon Kumara"},
]

nomes = [p["nome"] for p in produtos]
embeddings = model.encode(nomes)
sim_matrix = cosine_similarity(embeddings)

THRESHOLD = 0.85  # Limiar para considerar duplicata

print("🔍 DETECTOR DE PRODUTOS DUPLICADOS")
print(f"Threshold de similaridade: {THRESHOLD}")
print("="*60)

grupos_duplicatas = []
visitados = set()

for i in range(len(produtos)):
    if i in visitados:
        continue
    
    grupo = [i]
    for j in range(i + 1, len(produtos)):
        if j not in visitados and sim_matrix[i][j] >= THRESHOLD:
            grupo.append(j)
            visitados.add(j)
    
    if len(grupo) > 1:
        visitados.add(i)
        grupos_duplicatas.append(grupo)
        
        print(f"\n⚠️  POSSÍVEL DUPLICATA (similaridade ≥ {THRESHOLD}):")
        for idx in grupo:
            print(f"  ID {produtos[idx]['id']}: {produtos[idx]['nome']}")
        
        # Similaridade entre os itens do grupo
        if len(grupo) == 2:
            print(f"  Similaridade: {sim_matrix[grupo[0]][grupo[1]]:.3f}")

print(f"\n\n📊 RESUMO:")
print(f"  Total de produtos: {len(produtos)}")
print(f"  Grupos com possíveis duplicatas: {len(grupos_duplicatas)}")
print(f"  Produtos únicos estimados: {len(produtos) - sum(len(g)-1 for g in grupos_duplicatas)}")
```

---

## 🏆 Desafios Opcionais

1. **Visualização**: Use `matplotlib` para criar um mapa de calor da matriz de similaridade
2. **Clustering**: Use `KMeans` com embeddings para agrupar automaticamente textos por tema
3. **Multilingual**: Compare embeddings do mesmo texto em português e inglês
4. **Performance**: Meça o tempo de geração de embeddings local vs API para 100 textos

---

## ✅ Checklist de Entrega

- [ ] Ex01: comparação de similaridade funcionando
- [ ] Ex02: embeddings locais com sentence-transformers
- [ ] Ex03: buscador semântico com FAQ
- [ ] Detector de duplicatas funcionando
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 02](./pratica-02-prompt-engineering.md) | ➡️ **Próxima:** [Prática 04 — Banco Vetorial](./pratica-04-banco-vetorial.md)
