# Parte 05 — Agentes de IA

> **Carga horária:** 7 horas  
> **Práticas correspondentes:** [Prática 04](../praticas/pratica-04-agente-simples.md) e [Prática 05](../praticas/pratica-05-agente-ferramentas.md)

---

## 5.1 O que é um Agente de IA?

Um **agente de IA** é um sistema que usa um LLM como "cérebro" para:
1. **Perceber** o ambiente (através de inputs)
2. **Raciocinar** sobre o que fazer
3. **Agir** usando ferramentas
4. **Observar** os resultados
5. **Iterar** até completar a tarefa

> **Diferença chave:** Enquanto um chatbot simples responde a uma mensagem, um agente pode executar sequências complexas de ações para atingir um objetivo.

```
Chatbot: "Qual é o clima?" → "Não tenho acesso ao clima em tempo real."

Agente:  "Qual é o clima em Recife?" 
         → Pensa: "Preciso usar a ferramenta de clima"
         → Executa: get_weather("Recife")
         → Recebe: {"temp": 32, "condition": "ensolarado"}
         → Responde: "Está 32°C e ensolarado em Recife!"
```

---

## 5.2 O Loop ReAct

O padrão **ReAct (Reason + Act)** é a base de muitos agentes modernos:

```
Thought → Action → Observation → Thought → Action → ... → Final Answer
```

```
Thought: O usuário quer saber o clima e se precisa de guarda-chuva.
         Primeiro vou verificar o clima.
Action: get_weather(city="Recife")
Observation: {"temperature": 32, "condition": "sunny", "rain_chance": 5%}

Thought: Está ensolarado com 5% de chance de chuva. Não precisa de guarda-chuva.
Final Answer: Em Recife está 32°C e ensolarado. Você não precisará de guarda-chuva hoje.
```

---

## 5.3 Agente Simples com Function Calling

```python
from openai import OpenAI
import json
import requests

client = OpenAI()

# ============================================================
# DEFINIÇÃO DE FERRAMENTAS
# ============================================================

def search_web(query: str) -> str:
    """Simula busca na web."""
    # Em produção, use Tavily, SerpAPI, etc.
    return f"Resultados para '{query}': [artigo 1, artigo 2, artigo 3]"

def get_weather(city: str) -> dict:
    """Retorna dados meteorológicos."""
    # Simulated; em produção use OpenWeatherMap
    data = {
        "Recife": {"temp": 32, "condition": "ensolarado", "humidity": 70},
        "São Paulo": {"temp": 18, "condition": "nublado", "humidity": 85},
    }
    return data.get(city, {"error": "Cidade não encontrada"})

def calculate(expression: str) -> str:
    """Avalia expressão matemática de forma segura."""
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        return str(result)
    except Exception as e:
        return f"Erro: {e}"

# Mapeamento nome → função
TOOLS_MAP = {
    "search_web": search_web,
    "get_weather": get_weather,
    "calculate": calculate,
}

# Schema de ferramentas para a API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Busca informações na internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termos de busca"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtém clima atual de uma cidade",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nome da cidade"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calcula expressões matemáticas",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expressão matemática"}
                },
                "required": ["expression"]
            }
        }
    }
]

# ============================================================
# LOOP DO AGENTE
# ============================================================

def run_agent(user_message: str, max_steps: int = 10) -> str:
    """Executa o agente até completar a tarefa ou atingir o limite de passos."""
    
    messages = [
        {"role": "system", "content": "Você é um assistente útil com acesso a ferramentas. Use-as quando necessário para responder com precisão."},
        {"role": "user", "content": user_message}
    ]
    
    for step in range(max_steps):
        print(f"\n--- Passo {step + 1} ---")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # Agente terminou
        if finish_reason == "stop":
            print(f"✅ Resposta final: {message.content}")
            return message.content
        
        # Agente quer usar uma ferramenta
        if finish_reason == "tool_calls":
            messages.append(message)  # adiciona a mensagem do assistente com tool_calls
            
            # Executar cada ferramenta solicitada
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Chamando: {func_name}({func_args})")
                
                # Executar a função
                if func_name in TOOLS_MAP:
                    result = TOOLS_MAP[func_name](**func_args)
                else:
                    result = f"Erro: função {func_name} não encontrada"
                
                print(f"📥 Resultado: {result}")
                
                # Adicionar resultado ao histórico
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
    
    return "Limite de passos atingido sem resposta final."

# Testar
print(run_agent("Qual é o clima em Recife? Devo levar guarda-chuva?"))
print(run_agent("Quanto é 15% de 2750?"))
```

---

## 5.4 Function Calling em Detalhe

Function calling é o mecanismo que permite a um LLM **solicitar a execução de funções** no seu código. O modelo não executa nada diretamente — ele retorna um JSON estruturado indicando qual função quer chamar e com quais argumentos.

### 5.4.1 Anatomia de uma Tool Definition

Cada ferramenta é descrita por um **JSON Schema** que o modelo usa para entender quando e como chamá-la:

```python
tool_definition = {
    "type": "function",
    "function": {
        "name": "buscar_produto",            # nome único da função
        "description": "Busca produtos no catálogo por nome ou categoria. "
                       "Use quando o usuário perguntar sobre produtos disponíveis.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Termo de busca (nome ou categoria do produto)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Número máximo de resultados (padrão: 5)",
                    "default": 5
                },
                "ordem": {
                    "type": "string",
                    "enum": ["preco_asc", "preco_desc", "relevancia"],
                    "description": "Ordenação dos resultados"
                }
            },
            "required": ["query"]   # apenas query é obrigatório
        }
    }
}
```

> **💡 Dica:** A `description` da ferramenta é crucial — o modelo decide **quando** usar a ferramenta com base nela. Seja específico e inclua exemplos de quando usá-la.

### 5.4.2 Fluxo Completo de uma Tool Call

```python
import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 1. Definir ferramentas
tools = [
    {
        "type": "function",
        "function": {
            "name": "consultar_saldo",
            "description": "Consulta o saldo bancário de um cliente pelo CPF",
            "parameters": {
                "type": "object",
                "properties": {
                    "cpf": {"type": "string", "description": "CPF do cliente (apenas números)"}
                },
                "required": ["cpf"]
            }
        }
    }
]

# 2. Implementar a função real
def consultar_saldo(cpf: str) -> dict:
    saldos = {
        "12345678900": {"nome": "Maria", "saldo": 1500.00},
        "98765432100": {"nome": "João", "saldo": 320.50},
    }
    return saldos.get(cpf, {"erro": "CPF não encontrado"})

# 3. Enviar mensagem com ferramentas disponíveis
messages = [
    {"role": "user", "content": "Qual o saldo do CPF 12345678900?"}
]

response = client.chat.completions.create(
    model="llama3.2",
    messages=messages,
    tools=tools,
    tool_choice="auto"    # "auto", "none", ou {"type": "function", "function": {"name": "..."}}
)

msg = response.choices[0].message

# 4. Verificar se o modelo quer chamar uma ferramenta
if msg.tool_calls:
    # Adicionar a resposta do modelo ao histórico
    messages.append(msg)
    
    for tool_call in msg.tool_calls:
        nome = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # 5. Executar a função localmente
        resultado = consultar_saldo(**args)
        
        # 6. Retornar o resultado ao modelo
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(resultado, ensure_ascii=False)
        })
    
    # 7. Obter resposta final do modelo com o resultado da ferramenta
    response_final = client.chat.completions.create(
        model="llama3.2",
        messages=messages,
        tools=tools
    )
    print(response_final.choices[0].message.content)
```

### 5.4.3 Chamadas Paralelas de Ferramentas

O modelo pode solicitar **múltiplas ferramentas** em uma única resposta. Isso é útil quando várias informações independentes são necessárias:

```python
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def get_weather(city: str) -> dict:
    dados = {
        "Recife": {"temp": 32, "cond": "ensolarado"},
        "São Paulo": {"temp": 18, "cond": "chuvoso"},
        "Manaus": {"temp": 35, "cond": "parcialmente nublado"},
    }
    return dados.get(city, {"erro": "Cidade não encontrada"})

def get_population(city: str) -> dict:
    pops = {
        "Recife": 1_653_461,
        "São Paulo": 12_396_372,
        "Manaus": 2_255_903,
    }
    return {"city": city, "population": pops.get(city, "desconhecida")}

TOOLS_MAP = {"get_weather": get_weather, "get_population": get_population}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtém clima atual de uma cidade brasileira",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Obtém a população de uma cidade brasileira",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Compare clima e população de Recife e São Paulo"}
]

response = client.chat.completions.create(
    model="llama3.2", messages=messages, tools=tools
)

msg = response.choices[0].message

# O modelo pode retornar várias tool_calls de uma vez
if msg.tool_calls:
    print(f"Modelo solicitou {len(msg.tool_calls)} chamadas de ferramentas")
    messages.append(msg)
    
    # Executar todas as chamadas em paralelo
    def execute_tool(tc):
        func = TOOLS_MAP[tc.function.name]
        args = json.loads(tc.function.arguments)
        return tc.id, func(**args)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(execute_tool, msg.tool_calls))
    
    for call_id, result in results:
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False)
        })
    
    # Resposta final com todos os dados
    final = client.chat.completions.create(
        model="llama3.2", messages=messages, tools=tools
    )
    print(final.choices[0].message.content)
```

### 5.4.4 Function Calling com Ollama

O Ollama suporta function calling nativamente via API compatível com OpenAI. Basta apontar o `base_url` para o servidor local:

```python
from openai import OpenAI
import json

# Conexão com Ollama local — sem custo!
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "converter_moeda",
            "description": "Converte um valor de uma moeda para outra usando taxas fixas de exemplo",
            "parameters": {
                "type": "object",
                "properties": {
                    "valor": {"type": "number", "description": "Valor a converter"},
                    "de": {"type": "string", "description": "Moeda de origem (ex: BRL, USD, EUR)"},
                    "para": {"type": "string", "description": "Moeda de destino"}
                },
                "required": ["valor", "de", "para"]
            }
        }
    }
]

def converter_moeda(valor: float, de: str, para: str) -> dict:
    taxas = {
        ("BRL", "USD"): 0.20, ("USD", "BRL"): 5.00,
        ("BRL", "EUR"): 0.18, ("EUR", "BRL"): 5.50,
        ("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09,
    }
    taxa = taxas.get((de.upper(), para.upper()))
    if taxa is None:
        return {"erro": f"Conversão {de} → {para} não disponível"}
    return {"valor_original": valor, "de": de, "para": para, "resultado": round(valor * taxa, 2)}

messages = [{"role": "user", "content": "Quanto é 100 reais em dólares?"}]

response = client.chat.completions.create(
    model="llama3.2",
    messages=messages,
    tools=tools
)

msg = response.choices[0].message
if msg.tool_calls:
    messages.append(msg)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        resultado = converter_moeda(**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(resultado, ensure_ascii=False)
        })
    
    final = client.chat.completions.create(
        model="llama3.2", messages=messages, tools=tools
    )
    print(final.choices[0].message.content)
```

> **📌 Modelos com suporte a function calling no Ollama:** llama3.2, llama3.1, mistral, qwen2.5, command-r e outros. Verifique a [lista oficial](https://ollama.ai/search?c=tools) para modelos compatíveis.

---

## 5.5 Arquiteturas de Agentes

### Agente Único (Single Agent)

```
User → Agent (com ferramentas) → User
```
Simples e eficaz para tarefas moderadas.

### Multi-Agente

```
User → Orchestrator → [Specialist Agent 1]
                    → [Specialist Agent 2]
                    → [Specialist Agent 3]
       ← Orchestrator ←
User ←
```

Cada agente especializado em uma área (busca, código, análise de dados, etc.).

### Agente com Planejamento

```
User → Planner Agent → [Cria plano de subtarefas]
                     → Executor Agent 1 (subtarefa 1)
                     → Executor Agent 2 (subtarefa 2)
                     → ...
     ← Summarizer Agent ←
User ←
```

---

## 5.6 Memória em Agentes

Tipos de memória disponíveis:

```python
class AgentWithMemory:
    def __init__(self):
        self.client = OpenAI()
        
        # 1. Memória de conversação (curto prazo)
        self.conversation_history = []
        
        # 2. Memória episódica (longo prazo - armazenada em BD)
        self.episodic_memory = []  # em produção: banco vetorial
        
        # 3. Memória semântica (conhecimento - RAG)
        self.knowledge_base = None  # banco vetorial com docs
        
        # 4. Estado do agente
        self.state = {}
    
    def remember(self, key: str, value):
        """Salva informação importante no estado."""
        self.state[key] = value
    
    def recall(self, query: str) -> str:
        """Busca memórias relevantes."""
        # Em produção: busca semântica nas memórias episódicas
        relevant = [m for m in self.episodic_memory 
                    if query.lower() in m.lower()]
        return "\n".join(relevant[-3:]) if relevant else ""
```

---

## 5.7 Frameworks Open-Source para Agentes

Existem vários frameworks gratuitos e open-source para construir agentes. Abaixo, os mais relevantes para uso educacional.

### 5.7.1 smolagents (Hugging Face) — Recomendado para Iniciantes

O [smolagents](https://github.com/huggingface/smolagents) é a biblioteca de agentes da Hugging Face. É **100% open-source**, simples e funciona com qualquer modelo (local ou API).

```bash
pip install smolagents
```

```python
from smolagents import CodeAgent, tool, LiteLLMModel

# Usar modelo local via Ollama (gratuito!)
model = LiteLLMModel(model_id="ollama_chat/llama3.2")

# Definir uma ferramenta
@tool
def calcular_imc(peso_kg: float, altura_cm: float) -> str:
    """Calcula o IMC (Índice de Massa Corporal) de uma pessoa.
    
    Args:
        peso_kg: Peso em quilogramas.
        altura_cm: Altura em centímetros.
    """
    altura_m = altura_cm / 100
    imc = peso_kg / (altura_m ** 2)
    if imc < 18.5: cat = "abaixo do peso"
    elif imc < 25: cat = "peso normal"
    elif imc < 30: cat = "sobrepeso"
    else: cat = "obesidade"
    return f"IMC: {imc:.1f} — Categoria: {cat}"

# Criar e executar o agente
agent = CodeAgent(tools=[calcular_imc], model=model)
result = agent.run("Qual é o IMC de uma pessoa com 80kg e 1.75m?")
print(result)
```

> **💡 Vantagem:** smolagents gera código Python para resolver tarefas (CodeAgent), o que é mais transparente e educativo do que tool calling via JSON.

### 5.7.2 LangGraph — Agentes com Fluxo Controlado

LangGraph modela agentes como grafos com nós (ações) e arestas (condições):

```bash
pip install langgraph langchain-openai langchain-community
```

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int

def should_continue(state: AgentState) -> str:
    """Decide se continua ou para."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def call_model(state: AgentState) -> AgentState:
    """Nó: chama o LLM."""
    # Usar Ollama (gratuito) ou qualquer provedor compatível com OpenAI
    model = ChatOpenAI(
        model="llama3.2",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    response = model.invoke(state["messages"])
    return {"messages": [response], "step_count": state["step_count"] + 1}

def call_tools(state: AgentState) -> AgentState:
    """Nó: executa as ferramentas."""
    # ... executar tool calls
    return {"messages": [tool_result]}

# Construir o grafo
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

### 5.7.3 LangChain — Ecossistema Completo

LangChain oferece uma interface unificada para criar agentes com diversas ferramentas:

```bash
pip install langchain langchain-community langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Usar modelo local via Ollama (gratuito)
llm = ChatOpenAI(
    model="llama3.2",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

@tool
def buscar_cep(cep: str) -> str:
    """Busca informações de um CEP brasileiro."""
    import requests
    resp = requests.get(f"https://viacep.com.br/ws/{cep}/json/")
    if resp.status_code == 200:
        dados = resp.json()
        return f"{dados.get('logradouro', '')}, {dados.get('bairro', '')} - {dados.get('localidade', '')}/{dados.get('uf', '')}"
    return "CEP não encontrado"

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Use ferramentas quando necessário."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [buscar_cep], prompt)
executor = AgentExecutor(agent=agent, tools=[buscar_cep], verbose=True)

resultado = executor.invoke({"input": "Qual endereço do CEP 01001-000?"})
print(resultado["output"])
```

### Comparação de Frameworks

| Framework | Licença | Complexidade | Funciona com Ollama? | Melhor Para |
|-----------|---------|-------------|---------------------|-------------|
| **smolagents** | Apache 2.0 | Simples | ✅ | Aprendizado, agentes simples |
| **LangGraph** | MIT | Intermediária | ✅ | Fluxos complexos, multi-agente |
| **LangChain** | MIT | Intermediária | ✅ | Ecossistema completo, RAG + agentes |
| **CrewAI** | MIT | Simples | ✅ | Multi-agente com papéis definidos |

> **🎓 Recomendação para o curso:** Comece com **smolagents** (mais simples e educativo), depois avance para **LangGraph** quando precisar de fluxos mais complexos. Todos funcionam com **Ollama** (gratuito).

---

## 5.8 Ferramentas Comuns para Agentes

| Ferramenta | Uso | Biblioteca | Gratuita? |
|-----------|-----|-----------|-----------|
| Busca web | Informações atuais | DuckDuckGo Search, SearXNG | ✅ |
| Execução de código | Python, bash | subprocess, smolagents | ✅ |
| Busca em arquivos | PDFs, docs | LlamaIndex | ✅ |
| APIs externas | CEP, clima, dados | requests, httpx | ✅ |
| Banco de dados | SQL queries | SQLAlchemy | ✅ |
| Navegador web | Scraping, interação | Playwright, Selenium | ✅ |
| Cálculos | Matemática | Python stdlib | ✅ |

> **📌 Nota:** Todas as ferramentas listadas são gratuitas e open-source, adequadas para uso educacional.

---

## 5.9 Agentes Multi-Tool Avançados

Agentes reais costumam ter acesso a **várias ferramentas especializadas**. Nesta seção, construímos agentes mais sofisticados que combinam múltiplas capacidades.

### 5.9.1 Agente com RAG como Ferramenta

Em vez de usar RAG isoladamente, podemos integrá-lo como **uma ferramenta** de um agente. Assim, o agente decide quando consultar a base de conhecimento:

```python
from openai import OpenAI
import json
import numpy as np

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# --- Base de conhecimento simples (em produção: banco vetorial) ---
KNOWLEDGE_BASE = [
    {"id": 1, "content": "A política de devolução permite troca em até 30 dias com nota fiscal."},
    {"id": 2, "content": "O horário de atendimento é de segunda a sexta, das 8h às 18h."},
    {"id": 3, "content": "Frete grátis para compras acima de R$ 200,00 na região Sudeste."},
    {"id": 4, "content": "Parcelamento em até 12x sem juros no cartão de crédito."},
    {"id": 5, "content": "Garantia estendida disponível por R$ 49,90 para eletrônicos."},
]

def search_knowledge_base(query: str) -> str:
    """Busca na base de conhecimento por correspondência simples de palavras."""
    query_words = set(query.lower().split())
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((overlap, doc["content"]))
    scored.sort(reverse=True)
    if scored:
        return "\n".join(text for _, text in scored[:3])
    return "Nenhuma informação encontrada na base de conhecimento."

def get_order_status(order_id: str) -> dict:
    """Consulta status de um pedido."""
    orders = {
        "PED-001": {"status": "enviado", "previsao": "2025-01-20", "rastreio": "BR123456789"},
        "PED-002": {"status": "em separação", "previsao": "2025-01-22", "rastreio": None},
    }
    return orders.get(order_id, {"erro": "Pedido não encontrado"})

def calculate_shipping(cep: str, weight_kg: float) -> dict:
    """Calcula frete estimado."""
    base_cost = 15.0
    cost_per_kg = 2.5
    total = base_cost + (weight_kg * cost_per_kg)
    return {"cep": cep, "peso_kg": weight_kg, "frete": f"R$ {total:.2f}", "prazo": "5-7 dias úteis"}

TOOLS_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "get_order_status": get_order_status,
    "calculate_shipping": calculate_shipping,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Busca informações na base de conhecimento da empresa (políticas, FAQ, procedimentos). Use para dúvidas sobre regras e procedimentos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pergunta ou termos de busca"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Consulta o status de um pedido pelo ID (formato PED-XXX)",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "ID do pedido (ex: PED-001)"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": "Calcula o custo e prazo de frete para um CEP e peso",
            "parameters": {
                "type": "object",
                "properties": {
                    "cep": {"type": "string", "description": "CEP de destino"},
                    "weight_kg": {"type": "number", "description": "Peso em quilogramas"}
                },
                "required": ["cep", "weight_kg"]
            }
        }
    }
]

def run_support_agent(user_message: str, max_steps: int = 10) -> str:
    """Agente de suporte ao cliente com RAG + ferramentas."""
    messages = [
        {"role": "system", "content": (
            "Você é um agente de suporte ao cliente. Você tem acesso a:\n"
            "- Base de conhecimento da empresa (políticas, FAQ)\n"
            "- Sistema de consulta de pedidos\n"
            "- Calculadora de frete\n"
            "Use as ferramentas apropriadas para responder com precisão."
        )},
        {"role": "user", "content": user_message}
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="llama3.2", messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto"
        )
        msg = response.choices[0].message
        
        if response.choices[0].finish_reason == "stop":
            return msg.content
        
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                func = TOOLS_MAP.get(tc.function.name)
                args = json.loads(tc.function.arguments)
                result = func(**args) if func else {"erro": "Ferramenta não encontrada"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
                })
    
    return "Não foi possível resolver. Encaminhando para atendente humano."

# Exemplos de uso
print(run_support_agent("Qual é a política de devolução?"))
print(run_support_agent("Onde está meu pedido PED-001?"))
print(run_support_agent("Quanto custa o frete para CEP 50000-000, pacote de 3kg?"))
```

### 5.9.2 Agente de Pesquisa com Múltiplas Ferramentas

Um agente que busca informações, processa dados e gera relatórios:

```python
from openai import OpenAI
import json
from datetime import datetime

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# --- Ferramentas do agente de pesquisa ---

def search_web(query: str) -> str:
    """Simula busca na web (em produção: DuckDuckGo, SearXNG, Tavily)."""
    mock_results = {
        "PIB Brasil 2024": "O PIB do Brasil cresceu 3,1% em 2024, atingindo R$ 11,3 trilhões.",
        "população Brasil": "A população estimada do Brasil em 2024 é de 212 milhões de habitantes.",
        "inflação Brasil": "O IPCA acumulado em 2024 foi de 4,62%.",
    }
    for key, val in mock_results.items():
        if key.lower() in query.lower():
            return val
    return f"Resultados para '{query}': informação genérica encontrada."

def extract_data(text: str, fields: str) -> dict:
    """Extrai dados estruturados de um texto."""
    # Em produção: use o LLM para extração ou regex
    return {
        "texto_original": text[:200],
        "campos_solicitados": fields,
        "dados_extraidos": {"nota": "Dados extraídos com sucesso (simulado)"}
    }

def save_report(title: str, content: str) -> dict:
    """Salva um relatório (simulado — em produção: salva em arquivo ou BD)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "status": "salvo",
        "titulo": title,
        "timestamp": timestamp,
        "tamanho": f"{len(content)} caracteres"
    }

def calculate(expression: str) -> str:
    """Calcula uma expressão matemática."""
    try:
        allowed = {"__builtins__": {}}
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Erro: {e}"

TOOLS_MAP = {
    "search_web": search_web,
    "extract_data": extract_data,
    "save_report": save_report,
    "calculate": calculate,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Busca informações na internet sobre qualquer tema",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Termos de busca"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_data",
            "description": "Extrai dados estruturados de um texto bruto",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Texto de onde extrair dados"},
                    "fields": {"type": "string", "description": "Campos a extrair (ex: 'nome, data, valor')"}
                },
                "required": ["text", "fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Salva um relatório com título e conteúdo",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Título do relatório"},
                    "content": {"type": "string", "description": "Conteúdo completo do relatório"}
                },
                "required": ["title", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calcula expressões matemáticas",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Expressão matemática"}},
                "required": ["expression"]
            }
        }
    }
]

def run_research_agent(task: str, max_steps: int = 15) -> str:
    """Agente de pesquisa que busca, analisa e gera relatórios."""
    messages = [
        {"role": "system", "content": (
            "Você é um agente de pesquisa. Para completar tarefas:\n"
            "1. Use search_web para buscar informações\n"
            "2. Use extract_data para estruturar dados encontrados\n"
            "3. Use calculate para fazer cálculos necessários\n"
            "4. Use save_report para salvar o relatório final\n"
            "Seja metódico: busque dados, analise e depois gere o relatório."
        )},
        {"role": "user", "content": task}
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="llama3.2", messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto"
        )
        msg = response.choices[0].message
        
        if response.choices[0].finish_reason == "stop":
            return msg.content
        
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                func = TOOLS_MAP.get(tc.function.name)
                args = json.loads(tc.function.arguments)
                print(f"  🔧 [{tc.function.name}] {args}")
                result = func(**args) if func else {"erro": "Ferramenta não encontrada"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                })
    
    return "Pesquisa incompleta — limite de passos atingido."

# Exemplo
print(run_research_agent(
    "Pesquise sobre o PIB e a população do Brasil em 2024 e calcule o PIB per capita. "
    "Salve um relatório com os resultados."
))
```

---

## 5.10 Debugging e Observabilidade de Agentes

Agentes autônomos podem ser difíceis de depurar. Logging e rastreamento adequados são essenciais.

### 5.10.1 Logging de Decisões e Tool Calls

```python
import logging
import json
from datetime import datetime
from openai import OpenAI

# Configurar logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_log.jsonl"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent")

class ObservableAgent:
    def __init__(self, model: str = "llama3.2"):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model = model
        self.trace = []  # histórico completo de execução
    
    def log_event(self, event_type: str, data: dict):
        """Registra um evento no trace do agente."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data
        }
        self.trace.append(entry)
        logger.info(json.dumps(entry, ensure_ascii=False))
    
    def run(self, user_message: str, tools: list, tools_map: dict, max_steps: int = 10) -> str:
        self.trace = []
        self.log_event("start", {"input": user_message})
        
        messages = [
            {"role": "system", "content": "Você é um assistente útil com acesso a ferramentas."},
            {"role": "user", "content": user_message}
        ]
        
        for step in range(max_steps):
            self.log_event("llm_call", {"step": step + 1, "message_count": len(messages)})
            
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto"
            )
            msg = response.choices[0].message
            
            if response.choices[0].finish_reason == "stop":
                self.log_event("finish", {"response": msg.content[:200], "total_steps": step + 1})
                return msg.content
            
            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    self.log_event("tool_call", {
                        "tool": tc.function.name,
                        "args": args
                    })
                    
                    try:
                        func = tools_map[tc.function.name]
                        result = func(**args)
                        self.log_event("tool_result", {
                            "tool": tc.function.name,
                            "result": str(result)[:500]
                        })
                    except Exception as e:
                        result = {"erro": str(e)}
                        self.log_event("tool_error", {
                            "tool": tc.function.name,
                            "error": str(e)
                        })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                    })
        
        self.log_event("timeout", {"max_steps": max_steps})
        return "Limite de passos atingido."
    
    def print_trace(self):
        """Exibe o trace completo de forma legível."""
        print("\n📋 Trace de Execução do Agente:")
        print("=" * 50)
        for entry in self.trace:
            ts = entry["timestamp"].split("T")[1][:8]
            etype = entry["type"]
            if etype == "tool_call":
                print(f"  [{ts}] 🔧 {entry['tool']}({entry['args']})")
            elif etype == "tool_result":
                print(f"  [{ts}] 📥 → {entry['result'][:100]}")
            elif etype == "tool_error":
                print(f"  [{ts}] ❌ Erro: {entry['error']}")
            elif etype == "finish":
                print(f"  [{ts}] ✅ Finalizado em {entry['total_steps']} passos")
            else:
                print(f"  [{ts}] {etype}: {json.dumps({k: v for k, v in entry.items() if k not in ('timestamp', 'type')}, ensure_ascii=False)[:100]}")
```

### 5.10.2 Modos de Falha Comuns

| Problema | Sintoma | Solução |
|----------|---------|---------|
| **Loop infinito** | Agente chama a mesma ferramenta repetidamente | Limitar `max_steps`, detectar repetições |
| **Ferramenta errada** | Agente escolhe ferramenta inadequada | Melhorar descrições das ferramentas |
| **Argumentos inválidos** | JSON mal-formado ou tipos errados | Validar argumentos antes de executar |
| **Alucinação de ferramentas** | Agente tenta chamar ferramenta inexistente | Verificar se `func_name in TOOLS_MAP` |
| **Perda de contexto** | Agente esquece informações anteriores | Resumir histórico, usar memória explícita |
| **Custo descontrolado** | Muitas chamadas ao LLM sem convergir | Definir limites de tokens e passos |

```python
# Exemplo: detectando loops no agente
def detect_loop(messages: list, window: int = 4) -> bool:
    """Detecta se o agente está em loop (mesmas tool calls repetidas)."""
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc.function.name}:{tc.function.arguments}")
    
    if len(tool_calls) < window:
        return False
    
    # Verifica se as últimas N chamadas são iguais
    recent = tool_calls[-window:]
    return len(set(recent)) == 1
```

### 5.10.3 Observabilidade com Langfuse

O [Langfuse](https://langfuse.com/) é uma plataforma open-source para monitorar e depurar aplicações com LLMs. Ele registra traces, custos e latências automaticamente:

```bash
pip install langfuse
```

```python
from langfuse.openai import OpenAI as LangfuseOpenAI
import os

# Configurar Langfuse (self-hosted ou cloud gratuito)
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"  # self-hosted
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."

# Substituir o client OpenAI pelo wrapper do Langfuse
client = LangfuseOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Toda chamada ao client será automaticamente rastreada no Langfuse
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Olá!"}],
    name="agent-step-1",     # nome do trace (aparece no dashboard)
    metadata={"agent": "suporte", "step": 1}  # metadados extras
)
```

> **💡 Dica:** O Langfuse pode ser executado localmente via Docker (`docker compose up`) e oferece um dashboard visual para analisar traces de agentes, incluindo árvore de chamadas, latência por passo e custo estimado.

---

## 5.11 Padrões Práticos para Produção

Ao colocar agentes em ambientes reais, precisamos de mecanismos de resiliência e controle de custos.

### 5.11.1 Timeout e Retry

```python
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def call_llm_with_retry(messages: list, tools: list = None,
                        max_retries: int = 3, timeout: float = 30.0) -> dict:
    """Chama o LLM com retry e timeout."""
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="llama3.2",
                messages=messages,
                tools=tools,
                timeout=timeout
            )
            elapsed = time.time() - start
            print(f"  ⏱️  LLM respondeu em {elapsed:.1f}s (tentativa {attempt + 1})")
            return response
        except Exception as e:
            wait_time = 2 ** attempt  # backoff exponencial: 1s, 2s, 4s
            print(f"  ⚠️  Erro (tentativa {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Falha após {max_retries} tentativas: {e}")
```

### 5.11.2 Fallback entre Modelos

```python
from openai import OpenAI

def create_client(provider: str) -> tuple:
    """Retorna (client, model) para o provedor especificado."""
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), "llama3.2"
    elif provider == "openai":
        return OpenAI(), "gpt-4o-mini"
    raise ValueError(f"Provedor desconhecido: {provider}")

def call_with_fallback(messages: list, tools: list = None,
                       providers: list = None) -> dict:
    """Tenta provedores em ordem até um funcionar."""
    if providers is None:
        providers = ["ollama", "openai"]  # local primeiro, cloud como fallback
    
    for provider in providers:
        try:
            client, model = create_client(provider)
            print(f"  🔄 Tentando {provider} ({model})...")
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools, timeout=30.0
            )
            print(f"  ✅ Sucesso com {provider}")
            return response
        except Exception as e:
            print(f"  ❌ {provider} falhou: {e}")
            continue
    
    raise RuntimeError("Todos os provedores falharam")
```

### 5.11.3 Human-in-the-Loop

Para operações sensíveis, o agente deve solicitar aprovação humana antes de executar:

```python
class HumanInTheLoopAgent:
    # Ações que sempre precisam de aprovação
    REQUIRES_APPROVAL = {"send_email", "delete_record", "execute_sql", "transfer_money"}
    
    # Ações de baixo risco que podem executar automaticamente
    AUTO_APPROVE = {"search_web", "calculate", "get_weather", "search_knowledge_base"}
    
    def execute_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name in self.AUTO_APPROVE:
            return self._run_tool(tool_name, args)
        
        if tool_name in self.REQUIRES_APPROVAL:
            print(f"\n{'='*50}")
            print(f"⚠️  APROVAÇÃO NECESSÁRIA")
            print(f"Ferramenta: {tool_name}")
            print(f"Argumentos: {json.dumps(args, indent=2, ensure_ascii=False)}")
            print(f"{'='*50}")
            
            # Em produção: enviar notificação (Slack, email, webhook)
            # e aguardar resposta assíncrona
            aprovado = input("Aprovar execução? (s/n): ").strip().lower() == "s"
            
            if aprovado:
                return self._run_tool(tool_name, args)
            else:
                return {"status": "rejeitado", "motivo": "Operação não aprovada pelo usuário"}
        
        # Ferramentas desconhecidas são bloqueadas por padrão
        return {"status": "bloqueado", "motivo": f"Ferramenta '{tool_name}' não está na lista de permitidas"}
    
    def _run_tool(self, tool_name: str, args: dict) -> dict:
        func = TOOLS_MAP.get(tool_name)
        if func:
            return func(**args)
        return {"erro": f"Ferramenta '{tool_name}' não implementada"}
```

### 5.11.4 Controle de Custos

```python
class CostAwareAgent:
    # Custo aproximado por 1K tokens (em USD)
    COST_PER_1K = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "llama3.2": {"input": 0.0, "output": 0.0},  # local = gratuito!
    }
    
    def __init__(self, model: str = "llama3.2", budget_usd: float = 1.0):
        self.model = model
        self.budget = budget_usd
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
    
    def track_usage(self, response) -> bool:
        """Registra uso de tokens e verifica se está dentro do orçamento."""
        usage = response.usage
        if usage:
            self.total_tokens["input"] += usage.prompt_tokens
            self.total_tokens["output"] += usage.completion_tokens
            
            costs = self.COST_PER_1K.get(self.model, {"input": 0, "output": 0})
            step_cost = (
                (usage.prompt_tokens / 1000) * costs["input"] +
                (usage.completion_tokens / 1000) * costs["output"]
            )
            self.total_cost += step_cost
        
        if self.total_cost >= self.budget:
            print(f"  💰 ORÇAMENTO EXCEDIDO: ${self.total_cost:.4f} / ${self.budget:.2f}")
            return False  # orçamento estourado
        return True
    
    def get_usage_report(self) -> str:
        return (
            f"📊 Uso total: {self.total_tokens['input']} tokens input, "
            f"{self.total_tokens['output']} tokens output\n"
            f"💰 Custo total: ${self.total_cost:.4f} / ${self.budget:.2f}\n"
            f"💡 Dica: Use Ollama (custo $0.00) para desenvolvimento e testes!"
        )
```

> **🎓 Dica para o curso:** Use **Ollama** durante o desenvolvimento e testes (custo zero). Reserve APIs pagas para demonstrações com modelos de maior capacidade, quando necessário.

---

## 5.12 Segurança em Agentes

Agentes com acesso a ferramentas reais precisam de guardrails:

```python
class SafeAgent:
    DANGEROUS_OPERATIONS = ["DELETE", "DROP", "rm -rf"]
    
    def execute_tool(self, tool_name: str, args: dict):
        # 1. Confirmar ações destrutivas
        if tool_name in ["delete_file", "send_email", "execute_sql"]:
            if not self._get_human_approval(tool_name, args):
                return "Operação cancelada pelo usuário"
        
        # 2. Detectar operações perigosas
        for arg_val in args.values():
            if any(op in str(arg_val).upper() for op in self.DANGEROUS_OPERATIONS):
                return "Operação bloqueada por segurança"
        
        # 3. Limite de tentativas
        if self.step_count > 20:
            return "Limite de segurança atingido"
        
        return self._execute(tool_name, args)
    
    def _get_human_approval(self, tool: str, args: dict) -> bool:
        print(f"\n⚠️  O agente quer executar: {tool}({args})")
        return input("Aprovar? (s/n): ").lower() == "s"
```

---

## 📌 Resumo da Parte 05

| Conceito | Descrição |
|----------|-----------|
| Agente | LLM + ferramentas + loop de execução |
| ReAct | Padrão: Pensar → Agir → Observar → Repetir |
| Function Calling | Mecanismo para LLMs solicitarem execução de funções |
| Tool Definitions | JSON Schema que descreve ferramentas para o modelo |
| Chamadas Paralelas | Modelo pode solicitar várias ferramentas de uma vez |
| Function Calling (Ollama) | Suporte nativo via API compatível com OpenAI |
| Multi-Agente | Múltiplos agentes especializados colaborando |
| RAG como Ferramenta | Base de conhecimento integrada ao agente |
| Agente Multi-Tool | Agente com várias ferramentas especializadas |
| Agente de Pesquisa | Busca, extrai dados e gera relatórios |
| Observabilidade | Logging, tracing e monitoramento de agentes |
| Langfuse | Plataforma open-source para monitorar LLM apps |
| Timeout/Retry | Estratégias de resiliência para chamadas ao LLM |
| Fallback | Troca automática de provedor em caso de falha |
| Human-in-the-Loop | Aprovação humana para ações sensíveis |
| Controle de Custos | Monitoramento de tokens e orçamento |
| Guardrails | Controles de segurança para agentes autônomos |
| smolagents | Framework da Hugging Face, simples e open-source |
| LangGraph | Framework para agentes com fluxo controlado (grafos) |
| LangChain | Ecossistema completo para LLMs, RAG e agentes |

---

## 🔗 Referências

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [smolagents — Hugging Face](https://github.com/huggingface/smolagents)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [CrewAI — Multi-Agent Framework](https://github.com/crewAIInc/crewAI)
- [Ollama — Modelos locais gratuitos](https://ollama.ai)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Langfuse — LLM Observability](https://langfuse.com/)

---

⬅️ **Anterior:** [Parte 04 — Embeddings, Vetores e RAG](./parte-04-embeddings-vetores-rag.md) | ➡️ **Próximo:** [Parte 06 — Ferramentas com IA](./parte-06-ferramentas-com-ia.md)
