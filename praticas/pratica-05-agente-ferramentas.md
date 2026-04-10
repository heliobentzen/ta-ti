# Prática 05 — Agente com Múltiplas Ferramentas Especializadas

> **Carga horária estimada:** 3 horas  
> **Conteúdo relacionado:** [Parte 05](../conteudo/parte-05-agentes-ia.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Criar um conjunto rico de ferramentas especializadas para agentes
- Implementar verificação de segurança nas ferramentas
- Construir um agente capaz de raciocinar sobre tarefas multi-etapas
- Integrar RAG como uma ferramenta do agente

---

## 🔧 Setup

```bash
pip install openai chromadb python-dotenv sentence-transformers requests
```

---

## 📝 Exercício 1 — Agente Personal Trainer

Crie `pratica08/ex01_agente_fitness.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import date

load_dotenv()
client = OpenAI()

# Base de dados simulada
BANCO = {
    "usuario": {"nome": "João", "peso_kg": 80, "altura_cm": 175, "idade": 30, "objetivo": "perda de peso"},
    "treinos": [],
    "refeicoes": [],
    "metas": {"calorias_dia": 2000, "proteina_g": 150, "agua_ml": 3000}
}

def calcular_imc(peso_kg: float, altura_cm: float) -> dict:
    altura_m = altura_cm / 100
    imc = peso_kg / (altura_m ** 2)
    
    if imc < 18.5: categoria = "Abaixo do peso"
    elif imc < 25: categoria = "Peso normal"
    elif imc < 30: categoria = "Sobrepeso"
    else: categoria = "Obesidade"
    
    return {"imc": round(imc, 1), "categoria": categoria}

def registrar_treino(tipo: str, duracao_min: int, intensidade: str) -> dict:
    calorias = {"leve": 5, "moderada": 8, "intensa": 12}.get(intensidade, 7) * duracao_min
    treino = {
        "data": date.today().isoformat(),
        "tipo": tipo,
        "duracao_min": duracao_min,
        "intensidade": intensidade,
        "calorias_estimadas": calorias
    }
    BANCO["treinos"].append(treino)
    return {**treino, "mensagem": "Treino registrado com sucesso!"}

def gerar_treino(nivel: str, objetivo: str, disponibilidade_min: int) -> dict:
    treinos = {
        "perda de peso": {
            "aquecimento": "10 min de cardio leve",
            "principal": ["30 min corrida moderada", "3x15 agachamento", "3x15 burpee", "3x20 polichinelo"],
            "cooldown": "5 min alongamento"
        },
        "ganho de massa": {
            "aquecimento": "5 min aquecimento articular",
            "principal": ["4x8 supino", "4x8 remada", "4x10 desenvolvimento", "4x10 rosca"],
            "cooldown": "10 min alongamento"
        }
    }
    
    plano = treinos.get(objetivo, treinos["perda de peso"])
    return {
        "nivel": nivel,
        "objetivo": objetivo,
        "duracao_estimada": disponibilidade_min,
        "treino": plano
    }

def progresso_semana() -> dict:
    treinos_semana = BANCO["treinos"][-7:]  # últimos 7 registros como proxy
    calorias_total = sum(t["calorias_estimadas"] for t in treinos_semana)
    return {
        "treinos_realizados": len(treinos_semana),
        "calorias_totais_queimadas": calorias_total,
        "meta_treinos_semana": 4,
        "percentual_meta": min(100, len(treinos_semana) / 4 * 100)
    }

def dica_nutricional(objetivo: str) -> str:
    dicas = {
        "perda de peso": "Mantenha déficit calórico de 300-500 kcal/dia. Priorize proteínas magras e vegetais.",
        "ganho de massa": "Superávit de 300-500 kcal. Consuma 1.8-2.2g proteína por kg de peso corporal.",
        "manutenção": "Coma na manutenção calórica. Varie alimentos para garantir todos os micronutrientes."
    }
    return dicas.get(objetivo, "Consulte um nutricionista para orientação personalizada.")

TOOLS = [
    {"type": "function", "function": {
        "name": "calcular_imc",
        "description": "Calcula o IMC e categoria",
        "parameters": {"type": "object", "properties": {
            "peso_kg": {"type": "number"},
            "altura_cm": {"type": "number"}
        }, "required": ["peso_kg", "altura_cm"]}
    }},
    {"type": "function", "function": {
        "name": "registrar_treino",
        "description": "Registra um treino realizado",
        "parameters": {"type": "object", "properties": {
            "tipo": {"type": "string", "description": "ex: corrida, musculação, yoga"},
            "duracao_min": {"type": "integer"},
            "intensidade": {"type": "string", "enum": ["leve", "moderada", "intensa"]}
        }, "required": ["tipo", "duracao_min", "intensidade"]}
    }},
    {"type": "function", "function": {
        "name": "gerar_treino",
        "description": "Gera um plano de treino personalizado",
        "parameters": {"type": "object", "properties": {
            "nivel": {"type": "string", "enum": ["iniciante", "intermediario", "avancado"]},
            "objetivo": {"type": "string"},
            "disponibilidade_min": {"type": "integer"}
        }, "required": ["nivel", "objetivo", "disponibilidade_min"]}
    }},
    {"type": "function", "function": {
        "name": "progresso_semana",
        "description": "Mostra o progresso da semana",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "dica_nutricional",
        "description": "Fornece dica nutricional para um objetivo",
        "parameters": {"type": "object", "properties": {
            "objetivo": {"type": "string"}
        }, "required": ["objetivo"]}
    }}
]

TOOLS_MAP = {
    "calcular_imc": calcular_imc,
    "registrar_treino": registrar_treino,
    "gerar_treino": gerar_treino,
    "progresso_semana": progresso_semana,
    "dica_nutricional": dica_nutricional,
}

def personal_trainer(msg: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"""Você é um personal trainer virtual motivador.
Usuário: {BANCO['usuario']['nome']}, {BANCO['usuario']['peso_kg']}kg, 
{BANCO['usuario']['altura_cm']}cm, objetivo: {BANCO['usuario']['objetivo']}.
Use as ferramentas para dar orientações baseadas em dados reais."""
        },
        {"role": "user", "content": msg}
    ]
    
    for _ in range(6):
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg_resp = resp.choices[0].message
        
        if resp.choices[0].finish_reason == "stop":
            return msg_resp.content
        
        messages.append(msg_resp)
        for tc in msg_resp.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOLS_MAP[tc.function.name](**args)
            print(f"  🔧 {tc.function.name} → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, 
                           "content": json.dumps(result, ensure_ascii=False)})
    
    return "Timeout."

# Testar
interacoes = [
    "Qual é meu IMC atual?",
    "Acabei de fazer 45 minutos de corrida em intensidade moderada. Registra pra mim.",
    "Me gera um treino para hoje, tenho 60 minutos disponíveis. Sou iniciante.",
    "Dá uma dica de nutrição para meu objetivo.",
]

for i in interacoes:
    print(f"\n💬 {i}")
    print(f"🏋️ {personal_trainer(i)}")
```

---

## 📝 Exercício 2 — Agente com RAG como Ferramenta

Crie `pratica08/ex02_agente_rag_tool.py`:

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# Configurar base de conhecimento
chroma = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
kb = chroma.create_collection("empresa_kb", embedding_function=ef)

kb.add(
    documents=[
        "Horário de funcionamento: Segunda a sexta, 8h às 18h. Sábados, 9h às 13h.",
        "Endereço: Rua das Palmeiras, 100, Boa Viagem, Recife-PE, CEP 51020-000.",
        "Plano Basic: R$29/mês, até 3 usuários, 10GB. Plano Pro: R$79/mês, até 20 usuários, 100GB.",
        "Política de cancelamento: até 7 dias após contratação para reembolso total.",
        "Suporte: chat (24/7), email (48h úteis), telefone (11 3000-0000, horário comercial).",
        "Integração com Google Drive, Dropbox, OneDrive disponível no plano Pro.",
        "API disponível no plano Pro com limite de 10.000 requisições/mês.",
    ],
    ids=[f"kb{i}" for i in range(7)]
)

# Ferramentas do agente
def buscar_informacao(consulta: str) -> str:
    """Busca informações na base de conhecimento da empresa."""
    results = kb.query(query_texts=[consulta], n_results=3)
    docs = results["documents"][0]
    return "\n".join(docs) if docs else "Informação não encontrada."

def verificar_disponibilidade(data: str, horario: str) -> dict:
    """Verifica disponibilidade para uma reunião."""
    # Simulado
    disponivel = hash(f"{data}{horario}") % 3 != 0  # 66% disponível
    return {
        "data": data,
        "horario": horario,
        "disponivel": disponivel,
        "proximo_disponivel": "amanhã às 14h" if not disponivel else None
    }

def criar_ticket_suporte(tipo: str, descricao: str, prioridade: str) -> dict:
    """Cria um ticket de suporte."""
    ticket_id = f"TK{hash(descricao) % 10000:04d}"
    return {
        "ticket_id": ticket_id,
        "tipo": tipo,
        "descricao": descricao,
        "prioridade": prioridade,
        "status": "aberto",
        "previsao_resposta": "2 horas" if prioridade == "alta" else "24 horas"
    }

TOOLS = [
    {"type": "function", "function": {
        "name": "buscar_informacao",
        "description": "Busca informações sobre a empresa, produtos e políticas",
        "parameters": {"type": "object", "properties": {
            "consulta": {"type": "string"}
        }, "required": ["consulta"]}
    }},
    {"type": "function", "function": {
        "name": "verificar_disponibilidade",
        "description": "Verifica disponibilidade de horário para reunião ou suporte",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string", "description": "Data no formato YYYY-MM-DD"},
            "horario": {"type": "string", "description": "Horário, ex: 14:00"}
        }, "required": ["data", "horario"]}
    }},
    {"type": "function", "function": {
        "name": "criar_ticket_suporte",
        "description": "Cria um ticket de suporte para problemas técnicos",
        "parameters": {"type": "object", "properties": {
            "tipo": {"type": "string", "enum": ["bug", "duvida", "feature", "cobranca"]},
            "descricao": {"type": "string"},
            "prioridade": {"type": "string", "enum": ["baixa", "media", "alta"]}
        }, "required": ["tipo", "descricao", "prioridade"]}
    }}
]

TOOLS_MAP = {
    "buscar_informacao": buscar_informacao,
    "verificar_disponibilidade": verificar_disponibilidade,
    "criar_ticket_suporte": criar_ticket_suporte,
}

def atendente_virtual(pergunta: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """Você é um atendente virtual amigável e profissional.
Use buscar_informacao para consultar políticas e informações da empresa.
Abra tickets apenas quando o cliente tiver um problema técnico.
Sempre confirme o que foi feito ao final."""
        },
        {"role": "user", "content": pergunta}
    ]
    
    for _ in range(6):
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        
        if resp.choices[0].finish_reason == "stop":
            return msg.content
        
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOLS_MAP[tc.function.name](**args)
            print(f"  🔧 {tc.function.name}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                           "content": json.dumps(result, ensure_ascii=False)})
    
    return "Timeout."

# Cenários de atendimento
cenarios = [
    "Qual é o horário de atendimento de vocês?",
    "Quero cancelar meu plano, tenho direito a reembolso?",
    "Estou tendo um problema técnico grave, o sistema não carrega.",
    "Vocês têm integração com Google Drive?",
]

for c in cenarios:
    print(f"\n{'='*55}")
    print(f"👤 {c}")
    print(f"🤖 {atendente_virtual(c)}")
```

---

## 📝 Exercício 3 — Agente Pesquisador com Multi-Step Reasoning

Crie `pratica08/agente_pesquisador.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# Simular ferramentas de pesquisa
BASE_DADOS = {
    "python": {
        "criacao": 1991, "criador": "Guido van Rossum",
        "paradigmas": ["OOP", "funcional", "imperativo"],
        "casos_uso": ["IA/ML", "web", "automação", "ciência de dados"]
    },
    "javascript": {
        "criacao": 1995, "criador": "Brendan Eich",
        "paradigmas": ["OOP", "funcional", "event-driven"],
        "casos_uso": ["web frontend", "web backend (Node.js)", "mobile"]
    },
    "rust": {
        "criacao": 2010, "criador": "Graydon Hoare",
        "paradigmas": ["sistemas", "funcional", "concorrente"],
        "casos_uso": ["sistemas operacionais", "WebAssembly", "embedded"]
    }
}

def buscar_linguagem(nome: str) -> dict:
    """Busca informações sobre uma linguagem de programação."""
    nome_lower = nome.lower()
    return BASE_DADOS.get(nome_lower, {"erro": f"'{nome}' não encontrado na base"})

def comparar_linguagens(lang1: str, lang2: str, criterio: str) -> dict:
    """Compara duas linguagens por um critério."""
    info1 = BASE_DADOS.get(lang1.lower(), {})
    info2 = BASE_DADOS.get(lang2.lower(), {})
    
    return {
        "linguagem_1": lang1,
        "linguagem_2": lang2,
        "criterio": criterio,
        "info_1": info1,
        "info_2": info2
    }

def sugerir_linguagem(caso_uso: str, experiencia: str) -> dict:
    """Sugere a melhor linguagem para um caso de uso."""
    sugestoes = {
        "ia": {"principal": "Python", "razao": "Ecossistema rico: TensorFlow, PyTorch, scikit-learn"},
        "web frontend": {"principal": "JavaScript/TypeScript", "razao": "Único que roda no navegador nativamente"},
        "sistemas": {"principal": "Rust", "razao": "Performance e segurança de memória sem GC"},
        "backend": {"principal": "Python (FastAPI) ou JavaScript (Node.js)", "razao": "Rapidez de desenvolvimento e ecosistema"},
    }
    
    resultado = None
    for chave, val in sugestoes.items():
        if chave in caso_uso.lower():
            resultado = val
            break
    
    if not resultado:
        resultado = {"principal": "Python", "razao": "Linguagem versátil para iniciantes e experts"}
    
    resultado["experiencia"] = experiencia
    return resultado

TOOLS = [
    {"type": "function", "function": {
        "name": "buscar_linguagem",
        "description": "Busca informações sobre uma linguagem de programação",
        "parameters": {"type": "object", "properties": {
            "nome": {"type": "string"}
        }, "required": ["nome"]}
    }},
    {"type": "function", "function": {
        "name": "comparar_linguagens",
        "description": "Compara duas linguagens de programação",
        "parameters": {"type": "object", "properties": {
            "lang1": {"type": "string"},
            "lang2": {"type": "string"},
            "criterio": {"type": "string", "description": "O que comparar: idade, paradigmas, etc."}
        }, "required": ["lang1", "lang2", "criterio"]}
    }},
    {"type": "function", "function": {
        "name": "sugerir_linguagem",
        "description": "Sugere a melhor linguagem para um caso de uso",
        "parameters": {"type": "object", "properties": {
            "caso_uso": {"type": "string"},
            "experiencia": {"type": "string", "enum": ["iniciante", "intermediario", "experiente"]}
        }, "required": ["caso_uso", "experiencia"]}
    }}
]

TOOLS_MAP = {
    "buscar_linguagem": buscar_linguagem,
    "comparar_linguagens": comparar_linguagens,
    "sugerir_linguagem": sugerir_linguagem,
}

def pesquisador(pergunta: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """Você é um especialista em linguagens de programação.
Pesquise as informações necessárias antes de responder.
Dê respostas completas e fundamentadas nos dados reais."""
        },
        {"role": "user", "content": pergunta}
    ]
    
    for step in range(8):
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        
        if resp.choices[0].finish_reason == "stop":
            return msg.content
        
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOLS_MAP[tc.function.name](**args)
            print(f"  Step {step+1}: {tc.function.name}({args})")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                           "content": json.dumps(result, ensure_ascii=False)})
    
    return "Timeout."

# Testar raciocínio multi-step
perguntas_complexas = [
    "Python e JavaScript: qual devo aprender primeiro sendo iniciante?",
    "Quero construir um sistema de IA. Compare as opções disponíveis e me dê uma recomendação.",
    "Qual linguagem é mais moderna: Python, JavaScript ou Rust? Me explica a história delas.",
]

for p in perguntas_complexas:
    print(f"\n{'='*60}")
    print(f"❓ {p}\n")
    resposta = pesquisador(p)
    print(f"\n💬 {resposta}")
```

---

## 🏆 Desafios Opcionais

1. **Agente de email**: Crie ferramentas que simulam ler/enviar emails
2. **Rate limiting**: Adicione controle de taxa de chamadas às ferramentas
3. **Paralelismo**: Implemente chamadas paralelas a múltiplas ferramentas quando possível
4. **Fallback**: Se uma ferramenta falhar, o agente deve tentar uma alternativa

---

## ✅ Checklist de Entrega

- [ ] Ex01: agente personal trainer com 5+ ferramentas
- [ ] Ex02: agente com RAG integrado como ferramenta
- [ ] Ex03: agente pesquisador com raciocínio multi-step
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 04](./pratica-04-agente-simples.md) | ➡️ **Próxima:** [Prática 06 — Pipeline e Projeto Final](./pratica-06-pipeline-projeto-final.md)
