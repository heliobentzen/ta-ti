# Prática 02 — LLMs e Consumo de Contexto

> **Carga horária estimada:** 3 horas  
> **Conteúdo relacionado:** [Parte 02](../conteudo/parte-02-llms-consumo-contexto.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Contar tokens e entender como eles se traduzem em custo
- Dimensionar um orçamento de janela de contexto para diferentes casos de uso
- Escolher o modelo adequado baseando-se em tamanho de contexto e custo
- Medir e otimizar o consumo de tokens em chamadas reais à API

---

## 🔧 Configuração do Ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install openai tiktoken python-dotenv
```

Crie um arquivo `.env`:
```env
OPENAI_API_KEY=sk-...sua-chave-aqui...
```

> ⚠️ **NUNCA** commite o arquivo `.env` no repositório! Adicione ao `.gitignore`.

---

## 📝 Exercício 1 — Contando Tokens com tiktoken

Crie `pratica02/ex01_contar_tokens.py`:

```python
import tiktoken

def contar_tokens(texto: str, modelo: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(modelo)
    tokens = enc.encode(texto)
    return len(tokens)

def analisar_texto(texto: str, modelo: str = "gpt-4o-mini"):
    n_tokens = contar_tokens(texto, modelo)
    n_chars = len(texto)
    n_palavras = len(texto.split())
    
    print(f"Texto: {texto[:80]}{'...' if len(texto) > 80 else ''}")
    print(f"  Caracteres : {n_chars}")
    print(f"  Palavras   : {n_palavras}")
    print(f"  Tokens     : {n_tokens}")
    print(f"  Ratio tok/palavra: {n_tokens/n_palavras:.2f}")
    print()

# Analise diferentes tipos de texto
textos = [
    "Olá, como vai você?",
    "Hello, how are you?",
    "私は元気です。",  # japonês
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "SELECT u.nome, COUNT(p.id) AS total FROM usuarios u JOIN pedidos p ON u.id = p.usuario_id GROUP BY u.id;",
    "Lorem ipsum dolor sit amet " * 20,  # texto repetitivo
]

for texto in textos:
    analisar_texto(texto)
```

**Questões para responder:**
1. Qual tipo de texto tem maior ratio de tokens por palavra: inglês, português ou código?
2. Por que texto japonês tem um ratio muito mais alto?
3. O que isso implica no custo de sistemas multilíngues?

---

## 📝 Exercício 2 — Orçamento de Contexto

Crie `pratica02/ex02_orcamento_contexto.py`:

```python
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODELO = "gpt-4o-mini"
LIMITE_CONTEXTO = 128_000  # tokens — janela do gpt-4o-mini

def contar_tokens_mensagens(mensagens: list, modelo: str = MODELO) -> int:
    """Conta tokens no formato de chat (inclui overhead de formatação)."""
    enc = tiktoken.encoding_for_model(modelo)
    total = 0
    for msg in mensagens:
        total += 4  # overhead por mensagem
        for key, value in msg.items():
            total += len(enc.encode(value))
    total += 2  # overhead do assistente
    return total

# Simule uma conversa longa
system_prompt = """Você é um assistente especializado em Python. 
Responda sempre com exemplos de código funcionais e explicações detalhadas."""

historico = []
perguntas = [
    "O que são list comprehensions?",
    "Como funcionam decorators?",
    "Explique generators e yield.",
    "O que é asyncio e quando usar?",
    "Como funciona o GIL do Python?",
]

respostas_simuladas = [
    "List comprehensions são uma forma concisa de criar listas. Exemplo: `[x**2 for x in range(10)]` cria uma lista com os quadrados de 0 a 9. É equivalente a um loop for, mas mais legível e geralmente mais eficiente. Você pode adicionar condições: `[x for x in range(20) if x % 2 == 0]`." * 3,
    "Decorators são funções que modificam o comportamento de outras funções. São aplicados com `@nome_do_decorator` antes da definição da função. Internamente, `@decorator` é equivalente a `funcao = decorator(funcao)`. Muito usados para logging, autenticação e cache." * 3,
    "Generators são funções que usam `yield` para produzir valores um por vez, sob demanda. Diferentemente de listas, não carregam tudo na memória. São ideais para sequências grandes ou infinitas. `next()` avança o generator ao próximo valor." * 3,
    "asyncio é a biblioteca padrão do Python para programação assíncrona. Use quando seu código passa muito tempo esperando I/O (rede, disco). `async def` declara uma coroutine, `await` pausa sua execução até um resultado estar pronto." * 3,
    "O GIL (Global Interpreter Lock) é um mutex que protege objetos Python de modificações concorrentes. Ele impede que múltiplas threads executem bytecode Python simultaneamente, o que simplifica a implementação do CPython mas limita paralelismo real em CPU-bound tasks." * 3,
]

print(f"{'Turno':<6} {'Tokens Input':<15} {'Tokens Output':<15} {'Total Acumulado':<18} {'% Janela':<10}")
print("-" * 65)

total_acumulado = 0

for i, (pergunta, resposta) in enumerate(zip(perguntas, respostas_simuladas)):
    enc = tiktoken.encoding_for_model(MODELO)
    
    msgs = [{"role": "system", "content": system_prompt}] + historico + [{"role": "user", "content": pergunta}]
    tokens_input = contar_tokens_mensagens(msgs)
    tokens_output = len(enc.encode(resposta))
    total_acumulado = tokens_input + tokens_output
    pct = (total_acumulado / LIMITE_CONTEXTO) * 100
    
    print(f"{i+1:<6} {tokens_input:<15} {tokens_output:<15} {total_acumulado:<18} {pct:.1f}%")
    
    historico.append({"role": "user", "content": pergunta})
    historico.append({"role": "assistant", "content": resposta})

print(f"\nLimite da janela de contexto: {LIMITE_CONTEXTO:,} tokens")
print(f"\n💡 Em qual turno o contexto começa a ficar preocupante (>50%)?")
```

---

## 📝 Exercício 3 — Seleção de Modelo por Caso de Uso

Para cada caso de uso abaixo, escolha o modelo mais adequado e justifique. Considere: **tamanho do contexto necessário**, **custo por token**, **velocidade de resposta** e **qualidade necessária**.

**Modelos disponíveis (valores ilustrativos):**

| Modelo | Contexto máx. | Input ($/1M tok) | Output ($/1M tok) | Velocidade |
|--------|--------------|-------------------|--------------------|---------   |
| GPT-4o-mini | 128K | $0.15 | $0.60 | Rápida |
| GPT-4o | 128K | $2.50 | $10.00 | Média |
| Claude 3.5 Sonnet | 200K | $3.00 | $15.00 | Média |
| Claude 3 Haiku | 200K | $0.25 | $1.25 | Muito rápida |
| Llama 3.2 3B (local) | 128K | $0.00 | $0.00 | Rápida (GPU) |

**Casos de uso:**

1. **Chatbot de FAQ** para e-commerce: responde perguntas simples sobre entregas, devoluções e produtos. Volume: 10.000 consultas/dia.

2. **Análise de contratos jurídicos**: lê contratos de 100+ páginas e identifica cláusulas de risco. Volume: 20 contratos/dia.

3. **Assistente de código** integrado a um IDE: sugere completions em tempo real enquanto o dev digita. Latência máxima aceitável: 200ms.

4. **Moderação de conteúdo**: classifica se posts de usuários violam políticas. Volume: 100.000 posts/dia.

5. **Geração de relatórios executivos**: lê dados de um data warehouse e escreve narrativas analíticas. Volume: 50 relatórios/mês.

---

## 📝 Exercício 4 — Medindo Consumo Real (Projeto Principal)

Crie `pratica02/ex04_medicao_consumo.py` — uma ferramenta que mede e relata o consumo real de tokens:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
client = OpenAI()

class MonitorTokens:
    def __init__(self, modelo: str = "gpt-4o-mini", budget_tokens: int = 10_000):
        self.modelo = modelo
        self.budget = budget_tokens
        self.total_input = 0
        self.total_output = 0
        self.chamadas = []
    
    def chat(self, mensagens: list, **kwargs) -> str:
        response = client.chat.completions.create(
            model=self.modelo,
            messages=mensagens,
            **kwargs
        )
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.chamadas.append({
            "timestamp": datetime.now().isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
        
        return response.choices[0].message.content
    
    def relatorio(self, preco_input: float = 0.15, preco_output: float = 0.60):
        """Gera relatório de consumo. Preços em $/1M tokens."""
        custo_input = (self.total_input / 1_000_000) * preco_input
        custo_output = (self.total_output / 1_000_000) * preco_output
        custo_total = custo_input + custo_output
        pct_budget = (self.total_input + self.total_output) / self.budget * 100
        
        print("\n===== RELATÓRIO DE CONSUMO =====")
        print(f"Modelo: {self.modelo}")
        print(f"Chamadas realizadas: {len(self.chamadas)}")
        print(f"Tokens de entrada : {self.total_input:,}")
        print(f"Tokens de saída   : {self.total_output:,}")
        print(f"Total de tokens   : {self.total_input + self.total_output:,}")
        print(f"Budget utilizado  : {pct_budget:.1f}% de {self.budget:,}")
        print(f"Custo estimado    : ${custo_total:.6f} USD")
        print(f"  (input: ${custo_input:.6f} | output: ${custo_output:.6f})")
        print("================================\n")

# Teste com diferentes tarefas
monitor = MonitorTokens(budget_tokens=5_000)

tarefas = [
    {
        "nome": "Resumo curto",
        "msgs": [
            {"role": "system", "content": "Você é um assistente conciso."},
            {"role": "user", "content": "Resuma em 2 frases o que é machine learning."}
        ]
    },
    {
        "nome": "Análise detalhada",
        "msgs": [
            {"role": "system", "content": "Você é um especialista em Python."},
            {"role": "user", "content": "Explique detalhadamente as diferenças entre listas, tuplas, sets e dicionários em Python. Inclua exemplos de uso para cada um e quando preferir cada estrutura."}
        ]
    },
    {
        "nome": "Geração de código",
        "msgs": [
            {"role": "system", "content": "Você é um desenvolvedor senior."},
            {"role": "user", "content": "Escreva uma função Python que implementa busca binária com tipagem estática e docstring completa."}
        ]
    },
]

for tarefa in tarefas:
    print(f"Executando: {tarefa['nome']}...")
    resposta = monitor.chat(tarefa["msgs"])
    print(f"Resposta ({len(resposta)} chars): {resposta[:100]}...\n")

monitor.relatorio()
```

**Questões para analisar:**
1. Qual tarefa consumiu mais tokens de saída? Por quê?
2. Qual a proporção input/output em cada tarefa?
3. Como você otimizaria o system prompt para reduzir custos sem perder qualidade?

---

## 🏆 Desafios Opcionais

1. **Otimizador de prompts:** Crie uma função que recebe um prompt e sugere como encurtá-lo sem perder informação essencial (use o próprio LLM para ajudar).

2. **Comparação de modelos:** Refaça o Exercício 4 com dois modelos diferentes (ex: `gpt-4o-mini` e um modelo via Ollama). Compare custo vs. qualidade das respostas.

3. **Simulador de custos mensais:** Crie uma calculadora interativa que pergunta: tipo de sistema, volume de consultas/dia, tamanho médio das mensagens, e retorna o custo mensal estimado para cada modelo disponível.

---

## ✅ Checklist de Entrega

- [ ] Exercício 1: análise de tokens por tipo de texto com respostas às questões
- [ ] Exercício 2: tabela de orçamento de contexto preenchida
- [ ] Exercício 3: tabela de seleção de modelo com justificativas
- [ ] Exercício 4: código funcionando com relatório de consumo gerado
- [ ] Arquivo `.env` no `.gitignore`
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 01 — Desenho de Sistemas](./pratica-01-desenho-de-sistemas.md) | ➡️ **Próxima:** [Prática 03 — Engenharia de Contexto I](./pratica-03-engenharia-de-contexto-1.md)
