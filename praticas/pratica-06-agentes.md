# Prática 06 — Agentes no Sistema

> **Carga horária estimada:** 3 horas  
> **Conteúdo relacionado:** [Parte 06](../conteudo/parte-06-agentes-no-sistema.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Entender o loop ReAct (Reason → Act → Observe)
- Criar ferramentas (tools) para agentes
- Implementar um agente funcional com Function Calling
- Controlar o número de passos e segurança do agente

---

## 🔧 Setup

```bash
pip install openai python-dotenv requests
```

---

## 📝 Exercício 1 — Primeira Ferramenta

Crie `pratica07/ex01_primeira_ferramenta.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json
import math

load_dotenv()
client = OpenAI()

# ─────────────────────────────────────
# DEFINIR FERRAMENTAS
# ─────────────────────────────────────

def calcular(operacao: str, a: float, b: float) -> float:
    """Realiza operações matemáticas."""
    ops = {
        "soma": a + b,
        "subtracao": a - b,
        "multiplicacao": a * b,
        "divisao": a / b if b != 0 else "Erro: divisão por zero",
        "potencia": a ** b,
        "raiz": math.sqrt(a) if a >= 0 else "Erro: raiz de número negativo",
    }
    return ops.get(operacao, f"Operação '{operacao}' não suportada")

def converter_temperatura(valor: float, de: str, para: str) -> float:
    """Converte entre Celsius, Fahrenheit e Kelvin."""
    # Primeiro converter para Celsius
    if de == "fahrenheit":
        celsius = (valor - 32) * 5/9
    elif de == "kelvin":
        celsius = valor - 273.15
    else:
        celsius = valor
    
    # Converter de Celsius para destino
    if para == "fahrenheit":
        return celsius * 9/5 + 32
    elif para == "kelvin":
        return celsius + 273.15
    else:
        return celsius

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calcular",
            "description": "Realiza operações matemáticas básicas",
            "parameters": {
                "type": "object",
                "properties": {
                    "operacao": {
                        "type": "string",
                        "enum": ["soma", "subtracao", "multiplicacao", "divisao", "potencia", "raiz"],
                        "description": "Tipo de operação"
                    },
                    "a": {"type": "number", "description": "Primeiro número"},
                    "b": {"type": "number", "description": "Segundo número (ignorado para raiz)"}
                },
                "required": ["operacao", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "converter_temperatura",
            "description": "Converte temperaturas entre Celsius, Fahrenheit e Kelvin",
            "parameters": {
                "type": "object",
                "properties": {
                    "valor": {"type": "number"},
                    "de": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]},
                    "para": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]}
                },
                "required": ["valor", "de", "para"]
            }
        }
    }
]

TOOLS_MAP = {
    "calcular": calcular,
    "converter_temperatura": converter_temperatura
}

# ─────────────────────────────────────
# LOOP DO AGENTE
# ─────────────────────────────────────

def run_agent(user_message: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": "Você é um assistente matemático. Use as ferramentas disponíveis para cálculos."},
        {"role": "user", "content": user_message}
    ]
    
    for step in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        finish = response.choices[0].finish_reason
        
        if finish == "stop":
            return msg.content
        
        if finish == "tool_calls":
            if verbose:
                print(f"\n  [Passo {step+1}] Ferramenta(s) chamada(s):")
            
            messages.append(msg)
            
            for tc in msg.tool_calls:
                nome = tc.function.name
                args = json.loads(tc.function.arguments)
                
                if verbose:
                    print(f"    → {nome}({args})")
                
                resultado = TOOLS_MAP[nome](**args)
                
                if verbose:
                    print(f"    ← {resultado}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(resultado)
                })
    
    return "Não foi possível completar após 5 passos."

# Testar
perguntas = [
    "Quanto é 15 elevado a 3?",
    "Qual a raiz quadrada de 144?",
    "Converta 37°C para Fahrenheit",
    "Se eu tiver 250 e multiplicar por 0.15, qual é o resultado? E quanto sobra de 250?",
]

for p in perguntas:
    print(f"\n{'='*55}")
    print(f"❓ {p}")
    resposta = run_agent(p, verbose=True)
    print(f"\n💬 {resposta}")
```

---

## 📝 Exercício 2 — Agente com Ferramentas de Dados

Crie `pratica07/ex02_agente_dados.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import random

load_dotenv()
client = OpenAI()

# Simular banco de dados de vendas
VENDAS_DB = {
    "produtos": {
        "P001": {"nome": "Notebook Dell", "preco": 3500, "estoque": 15},
        "P002": {"nome": "Mouse Logitech", "preco": 150, "estoque": 80},
        "P003": {"nome": "Monitor LG 27'", "preco": 1800, "estoque": 23},
        "P004": {"nome": "Teclado Mecânico", "preco": 350, "estoque": 42},
        "P005": {"nome": "Headset Sony", "preco": 280, "estoque": 31},
    },
    "vendas": [
        {"data": "2024-01-15", "produto": "P001", "quantidade": 3, "valor": 10500},
        {"data": "2024-01-20", "produto": "P002", "quantidade": 10, "valor": 1500},
        {"data": "2024-02-05", "produto": "P003", "quantidade": 5, "valor": 9000},
        {"data": "2024-02-10", "produto": "P001", "quantidade": 2, "valor": 7000},
        {"data": "2024-02-15", "produto": "P004", "quantidade": 8, "valor": 2800},
        {"data": "2024-03-01", "produto": "P005", "quantidade": 12, "valor": 3360},
        {"data": "2024-03-10", "produto": "P002", "quantidade": 25, "valor": 3750},
    ]
}

# FERRAMENTAS
def listar_produtos() -> list:
    """Lista todos os produtos disponíveis."""
    return [
        {"id": pid, **info}
        for pid, info in VENDAS_DB["produtos"].items()
    ]

def verificar_estoque(produto_id: str) -> dict:
    """Verifica estoque de um produto."""
    produto = VENDAS_DB["produtos"].get(produto_id)
    if not produto:
        return {"erro": f"Produto {produto_id} não encontrado"}
    return {"produto_id": produto_id, "nome": produto["nome"], "estoque": produto["estoque"]}

def vendas_por_periodo(inicio: str, fim: str) -> dict:
    """Retorna resumo de vendas em um período."""
    vendas_filtradas = [
        v for v in VENDAS_DB["vendas"]
        if inicio <= v["data"] <= fim
    ]
    total = sum(v["valor"] for v in vendas_filtradas)
    return {
        "periodo": f"{inicio} a {fim}",
        "total_vendas": len(vendas_filtradas),
        "faturamento": total,
        "media_por_venda": total / len(vendas_filtradas) if vendas_filtradas else 0
    }

def produto_mais_vendido(mes: int = None) -> dict:
    """Retorna o produto mais vendido (no mês, se especificado)."""
    vendas = VENDAS_DB["vendas"]
    if mes:
        vendas = [v for v in vendas if int(v["data"].split("-")[1]) == mes]
    
    contagem = {}
    for v in vendas:
        pid = v["produto"]
        contagem[pid] = contagem.get(pid, 0) + v["quantidade"]
    
    if not contagem:
        return {"erro": "Sem vendas no período"}
    
    melhor = max(contagem, key=contagem.get)
    return {
        "produto_id": melhor,
        "nome": VENDAS_DB["produtos"][melhor]["nome"],
        "quantidade_vendida": contagem[melhor]
    }

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "listar_produtos",
            "description": "Lista todos os produtos disponíveis",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verificar_estoque",
            "description": "Verifica o estoque atual de um produto",
            "parameters": {
                "type": "object",
                "properties": {
                    "produto_id": {"type": "string", "description": "ID do produto (ex: P001)"}
                },
                "required": ["produto_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vendas_por_periodo",
            "description": "Retorna faturamento e número de vendas em um período",
            "parameters": {
                "type": "object",
                "properties": {
                    "inicio": {"type": "string", "description": "Data início (YYYY-MM-DD)"},
                    "fim": {"type": "string", "description": "Data fim (YYYY-MM-DD)"}
                },
                "required": ["inicio", "fim"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "produto_mais_vendido",
            "description": "Retorna o produto mais vendido",
            "parameters": {
                "type": "object",
                "properties": {
                    "mes": {"type": "integer", "description": "Mês (1-12). Opcional."}
                }
            }
        }
    }
]

TOOLS_MAP = {
    "listar_produtos": listar_produtos,
    "verificar_estoque": verificar_estoque,
    "vendas_por_periodo": vendas_por_periodo,
    "produto_mais_vendido": produto_mais_vendido,
}

def agente_vendas(pergunta: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente de análise de vendas. Use as ferramentas para consultar dados reais."
        },
        {"role": "user", "content": pergunta}
    ]
    
    for _ in range(6):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        msg = resp.choices[0].message
        
        if resp.choices[0].finish_reason == "stop":
            return msg.content
        
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOLS_MAP[tc.function.name](**args)
            print(f"  🔧 {tc.function.name}({args}) → {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
    
    return "Timeout do agente."

# Testar
perguntas_negocio = [
    "Quais produtos temos e qual o preço de cada um?",
    "Quanto faturamos em fevereiro de 2024?",
    "Qual foi o produto mais vendido em março?",
    "Compare o faturamento de janeiro com fevereiro de 2024.",
]

for p in perguntas_negocio:
    print(f"\n{'='*55}")
    print(f"❓ {p}")
    print(agente_vendas(p))
```

---

## 📝 Exercício 3 — Agente com Memória de Estado

Crie `pratica07/agente_com_memoria.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

class AgenteMemo:
    """Agente com estado persistente entre conversas."""
    
    def __init__(self):
        self.client = OpenAI()
        self.history = []
        self.memoria = {}  # memória de estado
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "salvar_na_memoria",
                    "description": "Salva uma informação importante para uso futuro",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chave": {"type": "string", "description": "Nome da informação"},
                            "valor": {"type": "string", "description": "Valor a salvar"}
                        },
                        "required": ["chave", "valor"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ler_da_memoria",
                    "description": "Lê uma informação salva anteriormente",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chave": {"type": "string"}
                        },
                        "required": ["chave"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "listar_memoria",
                    "description": "Lista todas as informações salvas na memória",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def _salvar_na_memoria(self, chave: str, valor: str) -> str:
        self.memoria[chave] = valor
        return f"Salvo: {chave} = {valor}"
    
    def _ler_da_memoria(self, chave: str) -> str:
        return self.memoria.get(chave, f"Chave '{chave}' não encontrada")
    
    def _listar_memoria(self) -> dict:
        return self.memoria if self.memoria else {"status": "Memória vazia"}
    
    def chat(self, mensagem: str) -> str:
        self.history.append({"role": "user", "content": mensagem})
        
        messages = [
            {
                "role": "system",
                "content": f"""Você é um assistente com memória persistente.
Use 'salvar_na_memoria' para guardar preferências e informações do usuário.
Use 'ler_da_memoria' quando precisar de informações guardadas anteriormente.

MEMÓRIA ATUAL: {json.dumps(self.memoria, ensure_ascii=False)}"""
            }
        ] + self.history
        
        for _ in range(5):
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            msg = resp.choices[0].message
            
            if resp.choices[0].finish_reason == "stop":
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content
            
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                nome = tc.function.name.replace("_da_", "_da_").lstrip("_")
                
                if tc.function.name == "salvar_na_memoria":
                    result = self._salvar_na_memoria(**args)
                elif tc.function.name == "ler_da_memoria":
                    result = self._ler_da_memoria(**args)
                else:
                    result = self._listar_memoria()
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        return "Timeout."

# Demonstração de memória persistente
agente = AgenteMemo()

conversas = [
    "Meu nome é Carlos e moro em Recife.",
    "Tenho preferência por respostas concisas e diretas.",
    "Qual é o meu nome?",
    "Em que cidade moro?",
    "Que preferências você tem registradas sobre mim?",
]

print("🧠 AGENTE COM MEMÓRIA\n")
for msg in conversas:
    print(f"👤 {msg}")
    resposta = agente.chat(msg)
    print(f"🤖 {resposta}")
    print(f"   [Memória atual: {agente.memoria}]")
    print()
```

---

## 🏆 Desafios Opcionais

1. **Agente Web**: Adicione uma ferramenta que faz requisições HTTP reais (ex: API de CEP)
2. **Limite de segurança**: Implemente um `max_cost` que encerra o agente se o custo estimado ultrapassar um limite
3. **Log de auditoria**: Registre todos os tool calls em um arquivo para auditoria
4. **Agente com aprovação**: Antes de executar certas ferramentas, peça confirmação ao usuário

---

## 📝 Exercício Bônus — Agente com smolagents (Open-Source)

O [smolagents](https://github.com/huggingface/smolagents) da Hugging Face é um framework simples e open-source para construir agentes. Diferente do Function Calling, o agente gera **código Python** para resolver tarefas, o que é mais transparente.

Crie `pratica07/bonus_smolagents.py`:

```bash
pip install smolagents litellm
```

```python
from smolagents import CodeAgent, tool, LiteLLMModel

# Usar modelo local via Ollama (gratuito!)
model = LiteLLMModel(model_id="ollama_chat/llama3.2")

# ─────────────────────────────────────
# DEFINIR FERRAMENTAS
# ─────────────────────────────────────

@tool
def calcular(operacao: str, a: float, b: float) -> str:
    """Realiza operações matemáticas básicas.
    
    Args:
        operacao: Tipo de operação ('soma', 'subtracao', 'multiplicacao', 'divisao', 'potencia').
        a: Primeiro número.
        b: Segundo número.
    """
    import math
    ops = {
        "soma": a + b,
        "subtracao": a - b,
        "multiplicacao": a * b,
        "divisao": a / b if b != 0 else "Erro: divisão por zero",
        "potencia": a ** b,
    }
    resultado = ops.get(operacao, f"Operação '{operacao}' não suportada")
    return f"{operacao}({a}, {b}) = {resultado}"

@tool
def converter_temperatura(valor: float, de: str, para: str) -> str:
    """Converte entre escalas de temperatura.
    
    Args:
        valor: Valor da temperatura.
        de: Escala de origem ('celsius', 'fahrenheit', 'kelvin').
        para: Escala de destino ('celsius', 'fahrenheit', 'kelvin').
    """
    # Converter para Celsius primeiro
    if de == "fahrenheit":
        celsius = (valor - 32) * 5/9
    elif de == "kelvin":
        celsius = valor - 273.15
    else:
        celsius = valor
    
    # Converter de Celsius para destino
    if para == "fahrenheit":
        resultado = celsius * 9/5 + 32
    elif para == "kelvin":
        resultado = celsius + 273.15
    else:
        resultado = celsius
    
    return f"{valor}° {de} = {resultado:.1f}° {para}"

@tool
def buscar_cep(cep: str) -> str:
    """Busca informações de endereço a partir de um CEP brasileiro.
    
    Args:
        cep: CEP no formato 'XXXXX-XXX' ou 'XXXXXXXX'.
    """
    import requests
    cep_limpo = cep.replace("-", "").strip()
    try:
        resp = requests.get(f"https://viacep.com.br/ws/{cep_limpo}/json/", timeout=5)
        if resp.status_code == 200:
            dados = resp.json()
            if "erro" in dados:
                return f"CEP {cep} não encontrado."
            return f"{dados.get('logradouro', '')}, {dados.get('bairro', '')} - {dados.get('localidade', '')}/{dados.get('uf', '')}"
    except Exception as e:
        return f"Erro ao buscar CEP: {e}"
    return "Falha na busca."

# ─────────────────────────────────────
# CRIAR E TESTAR O AGENTE
# ─────────────────────────────────────

agent = CodeAgent(
    tools=[calcular, converter_temperatura, buscar_cep],
    model=model,
    max_steps=5
)

perguntas = [
    "Quanto é 25 elevado a 3?",
    "Converta 100°F para Celsius.",
    "Qual o endereço do CEP 50030-230?",
]

for p in perguntas:
    print(f"\n{'='*55}")
    print(f"❓ {p}")
    resultado = agent.run(p)
    print(f"💬 {resultado}")
```

> **💡 Observe:** O smolagents mostra o código Python gerado pelo agente, permitindo entender exatamente como ele resolve cada tarefa. Isso é muito mais educativo do que o Function Calling opaco das APIs.

---

## ✅ Checklist de Entrega

- [ ] Ex01: agente com calculadora funcionando
- [ ] Ex02: agente de análise de vendas
- [ ] Ex03: agente com memória persistente
- [ ] Bônus: agente com smolagents (open-source, gratuito)
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 05](./pratica-05-rag.md) | ➡️ **Próxima:** [Prática 07 — Observabilidade](./pratica-07-observabilidade.md)
