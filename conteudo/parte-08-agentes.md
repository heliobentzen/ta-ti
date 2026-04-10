# Parte 08 — Agentes de IA

> **Carga horária:** 3 horas  
> **Práticas correspondentes:** [Prática 07](../praticas/pratica-07-agente-simples.md) e [Prática 08](../praticas/pratica-08-agente-ferramentas.md)

---

## 8.1 O que é um Agente de IA?

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

## 8.2 O Loop ReAct

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

## 8.3 Agente Simples com Function Calling

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

## 8.4 Arquiteturas de Agentes

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

## 8.5 Memória em Agentes

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

## 8.6 Frameworks Open-Source para Agentes

Existem vários frameworks gratuitos e open-source para construir agentes. Abaixo, os mais relevantes para uso educacional.

### 8.6.1 smolagents (Hugging Face) — Recomendado para Iniciantes

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

### 8.6.2 LangGraph — Agentes com Fluxo Controlado

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

### 8.6.3 LangChain — Ecossistema Completo

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

## 8.7 Ferramentas Comuns para Agentes

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

## 8.8 Segurança em Agentes

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

## 📌 Resumo da Parte 08

| Conceito | Descrição |
|----------|-----------|
| Agente | LLM + ferramentas + loop de execução |
| ReAct | Padrão: Pensar → Agir → Observar → Repetir |
| Function Calling | Mecanismo para LLMs solicitarem execução de funções |
| Multi-Agente | Múltiplos agentes especializados colaborando |
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

---

⬅️ **Anterior:** [Parte 07](./parte-07-rag.md) | ➡️ **Próximo:** [Parte 09 — Ferramentas com IA](./parte-09-ferramentas-com-ia.md)
