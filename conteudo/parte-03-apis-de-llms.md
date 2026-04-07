# Parte 03 — Trabalhando com APIs de LLMs

> **Carga horária:** 2 horas  
> **Prática correspondente:** [Prática 01](../praticas/pratica-01-primeiros-passos-llm.md) e [Prática 02](../praticas/pratica-02-prompt-engineering.md)

---

## 3.1 Visão Geral das APIs de LLM

As principais empresas de IA disponibilizam seus modelos via **APIs REST**. Isso significa que você não precisa instalar nada pesado — basta fazer uma requisição HTTP com sua chave de API.

### Padrão de Interface

A OpenAI popularizou um padrão de API que foi adotado (total ou parcialmente) por vários provedores:

```
POST /v1/chat/completions
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "Você é um assistente..."},
    {"role": "user", "content": "Qual é a capital do Brasil?"}
  ],
  "temperature": 0.7
}
```

---

## 3.2 Estrutura de Mensagens (Chat Format)

O formato de chat usa uma lista de mensagens com papéis distintos:

| Role | Descrição |
|------|-----------|
| `system` | Define o comportamento e persona do assistente |
| `user` | Mensagem do usuário |
| `assistant` | Resposta do modelo (histórico da conversa) |

```python
messages = [
    {"role": "system", "content": "Você é um especialista em Python."},
    {"role": "user", "content": "Como faço uma list comprehension?"},
    {"role": "assistant", "content": "Uma list comprehension tem a forma: [expr for item in iterable]"},
    {"role": "user", "content": "Pode dar um exemplo com filtro?"},
]
```

Para manter a conversa, você adiciona cada troca ao histórico e reenvia tudo.

---

## 3.3 API da OpenAI

### Configuração

```bash
pip install openai python-dotenv
```

```python
# .env
OPENAI_API_KEY=sk-...
```

### Exemplo Básico

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()  # lê OPENAI_API_KEY do ambiente

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Você é um assistente útil."},
        {"role": "user", "content": "Explique recursão em uma frase."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### Parâmetros Importantes

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `model` | str | ID do modelo |
| `messages` | list | Histórico de mensagens |
| `temperature` | float | Aleatoriedade (0–2) |
| `max_tokens` | int | Máximo de tokens na resposta |
| `top_p` | float | Nucleus sampling (0–1) |
| `frequency_penalty` | float | Penaliza repetição de tokens |
| `presence_penalty` | float | Penaliza tópicos já mencionados |
| `stream` | bool | Habilita streaming de tokens |

### Streaming

Para exibir respostas progressivamente (como o ChatGPT):

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Conte uma história curta."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 3.4 API da Anthropic (Claude)

```bash
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="Você é um assistente especializado em IA.",
    messages=[
        {"role": "user", "content": "O que é RAG?"}
    ]
)

print(message.content[0].text)
```

**Diferença chave**: Na API da Anthropic, o `system` é um parâmetro separado, não uma mensagem.

---

## 3.5 Modelos Open-Source com Ollama

Para usar modelos localmente sem custo por token:

```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Baixar e rodar um modelo
ollama pull llama3.2
ollama run llama3.2
```

```python
# Ollama tem API compatível com OpenAI
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # qualquer string
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Olá!"}]
)
print(response.choices[0].message.content)
```

---

## 3.6 Gerenciamento de Contexto e Memória

Por padrão, as APIs não mantêm histórico. Você deve gerenciar a memória manualmente:

```python
class Chatbot:
    def __init__(self, system_prompt: str):
        self.client = OpenAI()
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def clear_history(self):
        """Mantém apenas o system prompt"""
        self.messages = [self.messages[0]]

# Uso
bot = Chatbot("Você é um tutor de Python paciente e didático.")
print(bot.chat("O que são decorators?"))
print(bot.chat("Pode dar um exemplo?"))
```

### Estratégias para Contexto Longo

- **Janela deslizante**: mantém apenas as N últimas mensagens
- **Sumarização**: resume conversas antigas periodicamente
- **RAG**: busca apenas o contexto relevante (ver Parte 07)

---

## 3.7 Chamada de Funções (Function Calling / Tool Use)

Permite que o modelo solicite a execução de funções Python e use os resultados:

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtém a temperatura atual de uma cidade",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nome da cidade"}
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(city: str) -> str:
    # Em produção, chamaria uma API real
    return f"A temperatura em {city} é 28°C"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Qual é o tempo em Recife?"}],
    tools=tools,
    tool_choice="auto"
)

# Verificar se o modelo quer chamar uma função
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)
    
    # Executar a função
    result = get_weather(**func_args)
    print(f"Resultado da função: {result}")
```

---

## 3.8 Custos e Otimização

### Estrutura de Preços (exemplos aproximados)

| Modelo | Input (por 1M tokens) | Output (por 1M tokens) |
|--------|----------------------|----------------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |
| claude-3-haiku | $0.25 | $1.25 |
| claude-3-5-sonnet | $3.00 | $15.00 |

### Estratégias de Otimização de Custo

1. **Use o modelo certo para a tarefa**: tarefas simples → modelos menores
2. **Otimize prompts**: prompts menores = menos tokens de entrada
3. **Cache de respostas**: para perguntas frequentes e repetitivas
4. **Batch API**: a OpenAI oferece 50% de desconto para processamento em lote
5. **Modelos locais**: para desenvolvimento e prototipagem

---

## 3.9 Tratamento de Erros

```python
from openai import OpenAI, RateLimitError, APITimeoutError, APIError
import time

def resilient_call(client, messages, retries=3, delay=1):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except RateLimitError:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # exponential backoff
            else:
                raise
        except APITimeoutError:
            print(f"Timeout na tentativa {attempt + 1}")
            if attempt == retries - 1:
                raise
        except APIError as e:
            print(f"Erro da API: {e}")
            raise
```

---

## 📌 Resumo da Parte 03

| Conceito | Descrição |
|----------|-----------|
| Chat format | Mensagens com roles: system, user, assistant |
| Temperature | Controla criatividade da resposta |
| Streaming | Exibe tokens conforme gerados |
| Function Calling | Modelo pode requisitar execução de funções |
| Token cost | Cobrado por tokens de entrada + saída |
| Contexto manual | Developer gerencia histórico da conversa |

---

## 🔗 Referências

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Docs](https://docs.anthropic.com)
- [Ollama Documentation](https://ollama.ai/docs)
- [OpenAI Pricing](https://openai.com/pricing)

---

⬅️ **Anterior:** [Parte 02](./parte-02-llms-como-funcionam.md) | ➡️ **Próximo:** [Parte 04 — Prompt Engineering](./parte-04-prompt-engineering.md)
