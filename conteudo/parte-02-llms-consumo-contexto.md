# Parte 02 — LLMs e Consumo de Contexto

> **Carga horária:** 3h  
> **Prática correspondente:** [Prática 02](../praticas/pratica-02-llms-e-contexto.md)

---

## 2.1 O Token como Unidade Fundamental

Para trabalhar com LLMs profissionalmente, você precisa internalizar uma verdade simples: **tudo se reduz a tokens**. Tokens são a unidade de custo, a unidade de qualidade e a unidade de limite.

### O que é um Token?

Tokens não são palavras, não são caracteres, não são sílabas — são fragmentos de texto resultantes do processo de tokenização do modelo. Em português e inglês, uma regra de bolso razoável é:

- 1 token ≈ 4 caracteres em inglês
- 1 token ≈ 3-4 caracteres em português
- 1 palavra comum em inglês ≈ 1 token
- 1 palavra em português (língua com mais morfemas) ≈ 1.2-1.5 tokens
- Código Python ≈ menos tokens por caractere que texto em prosa

```
"Hello, world!" → ["Hello", ",", " world", "!"] → 4 tokens
"Olá, mundo!"   → ["Ol", "á", ",", " mundo", "!"] → 5 tokens
"def calcular_media(valores):" → tokens específicos de código → ~8 tokens
```

### Contando Tokens com tiktoken

A biblioteca `tiktoken` da OpenAI permite contar tokens antes de enviar ao modelo — essencial para controle de custo e planejamento de contexto:

```python
import tiktoken

def contar_tokens(texto: str, modelo: str = "gpt-4o") -> int:
    """Conta tokens para um modelo específico."""
    encoding = tiktoken.encoding_for_model(modelo)
    return len(encoding.encode(texto))

def analisar_prompt(mensagens: list[dict], modelo: str = "gpt-4o") -> dict:
    """
    Analisa o uso de tokens de uma lista de mensagens no formato chat.
    Inclui overhead de formato (tokens de metadados por mensagem).
    """
    encoding = tiktoken.encoding_for_model(modelo)
    overhead_por_mensagem = 4  # tokens de formato por mensagem
    overhead_resposta = 3      # tokens de prefixo da resposta

    total = overhead_resposta
    detalhes = []

    for msg in mensagens:
        tokens_conteudo = len(encoding.encode(msg["content"]))
        tokens_total_msg = tokens_conteudo + overhead_por_mensagem
        total += tokens_total_msg
        detalhes.append({
            "role": msg["role"],
            "tokens": tokens_total_msg
        })

    return {
        "total_tokens": total,
        "detalhes": detalhes
    }

# Exemplo de uso
mensagens = [
    {"role": "system", "content": "Você é um assistente especialista em Python."},
    {"role": "user", "content": "Como faço para ler um arquivo CSV em Python?"},
]

analise = analisar_prompt(mensagens)
print(f"Total de tokens na requisição: {analise['total_tokens']}")
for detalhe in analise['detalhes']:
    print(f"  {detalhe['role']}: {detalhe['tokens']} tokens")
```

### Tokens de Entrada vs. Tokens de Saída

Esta é uma distinção crítica que muitos desenvolvedores ignoram até receberem a primeira fatura:

**Tokens de saída custam de 3x a 5x mais que tokens de entrada.**

| Modelo | Entrada (por 1M tokens) | Saída (por 1M tokens) | Multiplicador |
|--------|------------------------|----------------------|---------------|
| GPT-4o | $5.00 | $15.00 | 3x |
| GPT-4o-mini | $0.15 | $0.60 | 4x |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 5x |
| Claude 3 Haiku | $0.25 | $1.25 | 5x |

**Implicação arquitetural:** Controlar o tamanho da saída (`max_tokens`) não é só uma questão de UX — é uma alavanca de custo direta. Um sistema que gera respostas verbosas desnecessariamente pode custar 3-5x mais do que deveria.

```python
# ❌ Caro: sem controle de saída
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explique machine learning"}]
    # Sem max_tokens → modelo pode gerar 2000+ tokens
)

# ✅ Econômico: saída controlada
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explique machine learning em 3 linhas"}],
    max_tokens=150  # Limite explícito
)
```

---

## 2.2 Janela de Contexto: O que de Fato Importa

### Contexto Não é Só Limite — é Fator de Qualidade

A janela de contexto é frequentemente explicada como "quanto o modelo consegue 'lembrar'". Isso é uma simplificação perigosa. O contexto não é memória — é a totalidade da informação disponível para o modelo processar em uma chamada.

Modelos modernos têm janelas enormes:
- GPT-4o: 128.000 tokens
- Claude 3.5 Sonnet: 200.000 tokens  
- Gemini 1.5 Pro: 1.000.000 tokens

**Mas isso não significa que você deve usá-las completamente.**

### O Problema do "Lost in the Middle"

Pesquisas empíricas (Liu et al., 2023) demonstraram um fenômeno importante: a performance dos LLMs **degrada sistematicamente** para informações posicionadas no meio de contextos longos.

```
Performance de recuperação de informação por posição no contexto:

Início do contexto  [████████████] Alta performance
Meio do contexto    [████░░░░████] Performance degradada
Final do contexto   [████████████] Alta performance
```

Implicação prática:
- Coloque instruções críticas e exemplos **no início** (system prompt)
- Coloque o documento ou dado mais relevante **no final** (logo antes da pergunta)
- Informações secundárias podem ficar no meio

```python
def montar_prompt_otimizado(
    instrucoes: str,
    contexto_secundario: str,
    documento_principal: str,
    pergunta: str
) -> list[dict]:
    """
    Monta mensagens com posicionamento estratégico de informações.
    Crítico vai no início (system) e imediatamente antes da pergunta.
    """
    return [
        {
            "role": "system",
            "content": instrucoes  # Alta atenção garantida aqui
        },
        {
            "role": "user",
            "content": (
                f"Contexto adicional:\n{contexto_secundario}\n\n"
                f"Documento principal:\n{documento_principal}\n\n"  # Alta atenção aqui também
                f"Pergunta: {pergunta}"
            )
        }
    ]
```

### Quando 128k Tokens se Torna um Problema

Usar toda a janela de contexto tem custos reais:

1. **Custo:** 128k tokens de entrada = custo substancial por chamada
2. **Latência:** Contextos maiores = mais tempo para processar
3. **Qualidade:** Degradação para informações no meio
4. **Atenção diluída:** O modelo "presta atenção" em tudo igualmente — contexto irrelevante polui a geração

**Regra prática:** Use o mínimo de contexto necessário para a tarefa. RAG existe exatamente para isso — em vez de colocar 1000 documentos no contexto, recupere os 3 mais relevantes.

---

## 2.3 Triângulo Velocidade-Custo-Qualidade

Em sistemas de LLM, você tem três dimensões de performance e só pode otimizar duas por vez. O modelo que você escolhe é a principal alavanca dessas dimensões.

```
         QUALIDADE
            /\
           /  \
          /    \
         /  ⚠️  \
        /  Escolha\
       /____________\
   VELOCIDADE    CUSTO
```

### Comparativo de Modelos (referência de mercado)

| Modelo | Velocidade (tok/s) | Custo relativo | Qualidade | Melhor para |
|--------|-------------------|----------------|-----------|-------------|
| GPT-4o | Média (60-80) | Alto | ⭐⭐⭐⭐⭐ | Tarefas complexas, raciocínio |
| GPT-4o-mini | Rápida (100-150) | Baixo | ⭐⭐⭐ | Volume alto, tarefas simples |
| Claude 3.5 Sonnet | Média-alta | Alto | ⭐⭐⭐⭐⭐ | Análise longa, código |
| Claude 3 Haiku | Muito rápida | Muito baixo | ⭐⭐⭐ | Classificação, extração |
| Llama 3.1 70B (local) | Depende da GPU | Infra fixa | ⭐⭐⭐⭐ | Volume alto, dados sensíveis |
| Llama 3.2 3B (local) | Muito rápida | Infra fixa | ⭐⭐ | Prototipagem, edge |

### TTFT vs. Tempo Total de Geração

**TTFT (Time to First Token):** Tempo entre o envio da requisição e o recebimento do primeiro token da resposta. É o que o usuário percebe como "lag" inicial.

**Tempo total:** TTFT + tempo de geração de todos os tokens.

```python
import time
from openai import OpenAI

client = OpenAI()

def medir_latencia_streaming(prompt: str, modelo: str = "gpt-4o-mini") -> dict:
    """Mede TTFT e tempo total com streaming."""
    inicio = time.perf_counter()
    primeiro_token_time = None
    tokens_gerados = 0

    with client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if primeiro_token_time is None:
                    primeiro_token_time = time.perf_counter()
                tokens_gerados += 1  # aproximação

    fim = time.perf_counter()

    ttft = (primeiro_token_time - inicio) * 1000 if primeiro_token_time else None
    total = (fim - inicio) * 1000

    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total, 1),
        "tokens_aproximados": tokens_gerados,
        "throughput_tok_s": round(tokens_gerados / (total / 1000), 1)
    }

# Uso
metricas = medir_latencia_streaming(
    "Explique o que é um token em LLMs",
    modelo="gpt-4o-mini"
)
print(f"TTFT: {metricas['ttft_ms']}ms")
print(f"Total: {metricas['total_ms']}ms")
print(f"Throughput: {metricas['throughput_tok_s']} tok/s")
```

### Quando Usar Qual Tier de Modelo

```
Complexidade da tarefa

ALTA │ GPT-4o / Claude 3.5 Sonnet
     │  ├─ Raciocínio multi-passo
     │  ├─ Análise de código complexo
     │  ├─ Geração de conteúdo longo e estruturado
     │  └─ Tarefas que precisam de "bom senso" amplo
     │
MÉDIA│ GPT-4o-mini / Claude 3 Sonnet
     │  ├─ Q&A com contexto
     │  ├─ Resumos
     │  ├─ Geração de código simples a médio
     │  └─ Extração estruturada
     │
BAIXA│ Claude 3 Haiku / GPT-4o-mini
     │  ├─ Classificação de texto
     │  ├─ Extração de entidades
     │  ├─ Reformatação
     │  └─ Roteamento (decidir qual pipeline usar)
     │
LOCAL│ Llama 3.x / Mistral (Ollama/vLLM)
     │  ├─ Dados sensíveis/confidenciais
     │  ├─ Volume muito alto (custo de API inviável)
     │  └─ Requisitos de latência ultra-baixa
```

---

## 2.4 Gerenciamento de Orçamento de Tokens

### Calculando o Uso Real de uma Conversa

Em aplicações de chat, o contexto cresce a cada turno — e o custo cresce junto:

```python
import tiktoken
from dataclasses import dataclass, field

@dataclass
class TokenBudgetManager:
    """
    Gerencia o orçamento de tokens de uma conversa, rastreando
    custos e alertando quando limites são atingidos.
    """
    modelo: str = "gpt-4o-mini"
    limite_contexto: int = 128_000
    reserva_saida: int = 2_000
    preco_entrada_por_milhao: float = 0.15   # USD, gpt-4o-mini
    preco_saida_por_milhao: float = 0.60      # USD, gpt-4o-mini

    _historico: list = field(default_factory=list)
    _tokens_entrada_total: int = field(default=0, init=False)
    _tokens_saida_total: int = field(default=0, init=False)

    def __post_init__(self):
        self._encoding = tiktoken.encoding_for_model(self.modelo)

    def _contar_tokens(self, texto: str) -> int:
        return len(self._encoding.encode(texto))

    def adicionar_mensagem(self, role: str, content: str) -> dict:
        """Adiciona mensagem e retorna análise de tokens."""
        tokens = self._contar_tokens(content) + 4  # overhead por mensagem
        self._historico.append({"role": role, "content": content})

        if role in ("user", "system"):
            self._tokens_entrada_total += tokens
        else:
            self._tokens_saida_total += tokens

        return {"tokens_mensagem": tokens, "status": self._verificar_limite()}

    def _verificar_limite(self) -> str:
        total = self._tokens_entrada_total + self._tokens_saida_total
        percentual = total / self.limite_contexto
        if percentual > 0.9:
            return "CRITICO"
        elif percentual > 0.7:
            return "ALERTA"
        return "OK"

    def custo_acumulado(self) -> dict:
        custo_entrada = (self._tokens_entrada_total / 1_000_000) * self.preco_entrada_por_milhao
        custo_saida = (self._tokens_saida_total / 1_000_000) * self.preco_saida_por_milhao
        return {
            "tokens_entrada": self._tokens_entrada_total,
            "tokens_saida": self._tokens_saida_total,
            "custo_entrada_usd": round(custo_entrada, 6),
            "custo_saida_usd": round(custo_saida, 6),
            "custo_total_usd": round(custo_entrada + custo_saida, 6),
        }

    def precisa_truncar(self) -> bool:
        total = self._tokens_entrada_total + self._tokens_saida_total
        return total > (self.limite_contexto - self.reserva_saida)

    def truncar_historico(self, manter_system: bool = True) -> None:
        """Remove mensagens mais antigas, preservando o system prompt."""
        if not self.precisa_truncar():
            return

        system_msgs = [m for m in self._historico if m["role"] == "system"]
        outras_msgs = [m for m in self._historico if m["role"] != "system"]

        # Remove mensagens mais antigas até caber no contexto
        while self.precisa_truncar() and len(outras_msgs) > 2:
            removida = outras_msgs.pop(0)
            tokens_removidos = self._contar_tokens(removida["content"]) + 4
            self._tokens_entrada_total -= tokens_removidos

        self._historico = system_msgs + outras_msgs if manter_system else outras_msgs


# Uso em produção
manager = TokenBudgetManager(modelo="gpt-4o-mini")

manager.adicionar_mensagem("system", "Você é um assistente de código Python.")
manager.adicionar_mensagem("user", "Como faço um loop em Python?")
manager.adicionar_mensagem("assistant", "Use `for item in lista:` para iterar...")
manager.adicionar_mensagem("user", "E como faço um loop com índice?")

custo = manager.custo_acumulado()
print(f"Tokens usados: {custo['tokens_entrada'] + custo['tokens_saida']}")
print(f"Custo acumulado: ${custo['custo_total_usd']:.6f} USD")
```

### Estratégias de Truncamento de Histórico

| Estratégia | Quando usar | Trade-off |
|------------|-------------|-----------|
| Sliding window | Chat genérico | Perde contexto antigo, simples |
| Summary compression | Conversas longas | Qualidade preservada, mais chamadas |
| Relevance-based pruning | Conversas com tópicos variados | Complexo, melhor resultado |
| System prompt fixo + últimas N | Assistente com personalidade definida | Simples, funciona bem na maioria |

---

## 2.5 Rate Limits e Planejamento de Throughput

### Entendendo os Limites

Provedores de LLM impõem dois tipos de rate limit:

- **RPM (Requests Per Minute):** Número de chamadas por minuto
- **TPM (Tokens Per Minute):** Tokens processados por minuto (entrada + saída)

O limite que você atinge primeiro depende do seu padrão de uso:

```
Se suas chamadas têm poucos tokens → você atinge RPM primeiro
Se suas chamadas têm muitos tokens → você atinge TPM primeiro
```

### Tiers da OpenAI (referência)

| Tier | RPM | TPM | Como alcançar |
|------|-----|-----|---------------|
| Tier 1 | 500 | 30.000 | $5 gastos |
| Tier 2 | 5.000 | 450.000 | $50 gastos |
| Tier 3 | 5.000 | 800.000 | $100 gastos |
| Tier 4 | 10.000 | 2.000.000 | $250 gastos |
| Tier 5 | 10.000 | 30.000.000 | $1.000 gastos |

### Implementando Exponential Backoff com Jitter

```python
import asyncio
import random
import time
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar("T")

class RateLimitHandler:
    """
    Handler de rate limit com exponential backoff e jitter.
    Jitter evita o "thundering herd problem" quando muitos
    workers tentam retry simultâneo.
    """

    def __init__(
        self,
        max_tentativas: int = 5,
        delay_base: float = 1.0,
        delay_maximo: float = 60.0,
        fator_multiplicador: float = 2.0,
    ):
        self.max_tentativas = max_tentativas
        self.delay_base = delay_base
        self.delay_maximo = delay_maximo
        self.fator_multiplicador = fator_multiplicador

    def calcular_delay(self, tentativa: int) -> float:
        """Exponential backoff com full jitter."""
        delay_exponencial = min(
            self.delay_base * (self.fator_multiplicador ** tentativa),
            self.delay_maximo
        )
        # Full jitter: aleatoriza entre 0 e o delay calculado
        return random.uniform(0, delay_exponencial)

    async def executar_com_retry(
        self,
        func: Callable,
        *args,
        erros_retriable: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Executa função assíncrona com retry automático em erros de rate limit.
        """
        ultimo_erro = None

        for tentativa in range(self.max_tentativas):
            try:
                return await func(*args, **kwargs)

            except erros_retriable as e:
                ultimo_erro = e
                erro_str = str(e).lower()

                # Verifica se é erro de rate limit
                eh_rate_limit = any(
                    termo in erro_str
                    for termo in ["rate limit", "429", "too many requests", "quota"]
                )

                if not eh_rate_limit or tentativa == self.max_tentativas - 1:
                    raise

                delay = self.calcular_delay(tentativa)
                print(
                    f"Rate limit atingido (tentativa {tentativa + 1}/{self.max_tentativas}). "
                    f"Aguardando {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise ultimo_erro


# Uso prático
handler = RateLimitHandler(max_tentativas=5, delay_base=1.0)

async def chamar_com_retry(prompt: str) -> str:
    async def _chamar():
        # Simulação — em produção, chame o SDK real
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    return await handler.executar_com_retry(_chamar)
```

### Planejamento de Throughput para Produção

```python
def calcular_throughput_maximo(
    rpm_disponivel: int,
    tokens_entrada_media: int,
    tokens_saida_media: int,
    tpm_disponivel: int
) -> dict:
    """Calcula throughput máximo considerando ambos os limites."""
    # Limite por RPM
    chamadas_por_rpm = rpm_disponivel

    # Limite por TPM
    tokens_por_chamada = tokens_entrada_media + tokens_saida_media
    chamadas_por_tpm = tpm_disponivel // tokens_por_chamada

    # O limite efetivo é o menor dos dois
    limite_efetivo = min(chamadas_por_rpm, chamadas_por_tpm)
    gargalo = "RPM" if chamadas_por_rpm < chamadas_por_tpm else "TPM"

    return {
        "chamadas_por_minuto_maximo": limite_efetivo,
        "chamadas_por_hora": limite_efetivo * 60,
        "chamadas_por_dia": limite_efetivo * 60 * 24,
        "gargalo": gargalo,
        "utilizacao_rpm": round(limite_efetivo / rpm_disponivel * 100, 1),
        "utilizacao_tpm": round(
            (limite_efetivo * tokens_por_chamada) / tpm_disponivel * 100, 1
        ),
    }

# Exemplo: Tier 2 da OpenAI com gpt-4o-mini
resultado = calcular_throughput_maximo(
    rpm_disponivel=5_000,
    tokens_entrada_media=800,
    tokens_saida_media=400,
    tpm_disponivel=450_000
)
print(resultado)
# gargalo: TPM, limite: 375 chamadas/minuto
```

---

## 2.6 Latência na Prática: TTFT vs. Tempo Total

### Por que TTFT Importa para UX

Em interfaces de chat ou ferramentas interativas, o usuário não espera a resposta completa para começar a ler — ele começa a ler no primeiro token. Um TTFT de 300ms com streaming faz um sistema de 10 segundos de geração total parecer rápido. Sem streaming, o usuário vê uma tela em branco por 10 segundos.

**Streaming é obrigatório para qualquer interface interativa.**

### Implementando Streaming Corretamente

```python
import sys
from openai import OpenAI

client = OpenAI()

def stream_resposta(prompt: str, modelo: str = "gpt-4o-mini") -> str:
    """
    Faz streaming da resposta, exibindo em tempo real.
    Retorna o texto completo ao final.
    """
    resposta_completa = []

    with client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)  # flush garante output imediato
                resposta_completa.append(delta)

    print()  # nova linha ao final
    return "".join(resposta_completa)


async def stream_para_websocket(prompt: str, ws) -> None:
    """
    Exemplo de streaming para WebSocket (padrão de produção).
    ws seria uma conexão WebSocket real (fastapi, django channels, etc.)
    """
    async with client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                await ws.send_text(delta)
        # Sinaliza fim da stream
        await ws.send_text("[DONE]")
```

### Batching para Throughput

Quando latência por item não é crítica (pipelines de dados, processamento noturno), agrupe chamadas para maximizar throughput:

```python
import asyncio
from typing import List

async def processar_lote_otimizado(
    textos: List[str],
    modelo: str = "gpt-4o-mini",
    concorrencia_maxima: int = 10,
    prompt_template: str = "Classifique o sentimento do texto: {texto}"
) -> List[str]:
    """
    Processa um lote de textos em paralelo com controle de concorrência.
    Ideal para pipelines de dados onde latência por item não é crítica.
    """
    semaforo = asyncio.Semaphore(concorrencia_maxima)
    resultados = [None] * len(textos)

    async def processar_item(idx: int, texto: str) -> None:
        async with semaforo:
            try:
                response = await client.chat.completions.acreate(
                    model=modelo,
                    messages=[{
                        "role": "user",
                        "content": prompt_template.format(texto=texto)
                    }],
                    max_tokens=50,
                )
                resultados[idx] = response.choices[0].message.content
            except Exception as e:
                resultados[idx] = f"ERRO: {e}"

    tarefas = [
        processar_item(idx, texto)
        for idx, texto in enumerate(textos)
    ]
    await asyncio.gather(*tarefas)
    return resultados
```

---

## 2.7 Seleção de Modelo por Requisito

### Framework de Decisão

Classificar sua tarefa é o primeiro passo para escolher o modelo certo:

```
PASSO 1: Qual é a natureza da tarefa?
  ├─ Classificação / extração → modelo pequeno pode resolver
  ├─ Resumo / reescrita → modelo médio geralmente suficiente
  ├─ Raciocínio / análise complexa → modelo grande necessário
  └─ Geração de código → depende da complexidade

PASSO 2: Qual é o requisito de latência?
  ├─ < 500ms TTFT (interativo) → streaming + modelo rápido
  ├─ < 5s total (operacional) → modelo médio
  └─ Sem requisito de tempo (batch) → maximize qualidade/custo

PASSO 3: Qual é o volume?
  ├─ < 1.000 chamadas/dia → qualquer modelo, custo irrelevante
  ├─ 1.000 - 100.000/dia → calcule custo, considere mini/haiku
  └─ > 100.000/dia → considere open source self-hosted

PASSO 4: Os dados são sensíveis?
  ├─ Sim → self-hosted obrigatório
  └─ Não → API comercial ok
```

### Tabela de Decisão Prática

| Tipo de Tarefa | Modelo Recomendado | Por quê |
|----------------|-------------------|---------|
| Classificação de sentimento | claude-3-haiku ou gpt-4o-mini | Tarefa simples, alta velocidade, baixo custo |
| Extração de entidades (NER) | gpt-4o-mini | Confiável para structured output |
| Resumo de documento (~5 páginas) | gpt-4o-mini | Suficiente para sumarização padrão |
| Resumo de documento (100+ páginas) | claude-3-5-sonnet | Janela grande, atenção a documentos longos |
| Geração de código simples | gpt-4o-mini | Suficiente para snippets e boilerplate |
| Code review / refactoring complexo | gpt-4o ou claude-3-5-sonnet | Raciocínio profundo necessário |
| Q&A sobre base de conhecimento (RAG) | gpt-4o-mini | RAG reduz necessidade do modelo grande |
| Raciocínio multi-passo (agente) | gpt-4o ou claude-3-5-sonnet | Necessita planejamento e tool use confiável |
| Análise de sentimento em escala (batch) | Llama 3.1 local | Volume justifica infra própria |
| Moderação de conteúdo | gpt-4o-mini | Alta precisão, latência aceitável, barato |

---

## 2.8 Modelos Open Source como Alternativa Real

### Quando Faz Sentido Usar Modelos Locais

O ecossistema open source de LLMs amadureceu significativamente. Modelos como Llama 3.1, Mistral, Qwen e Command R+ competem com modelos comerciais de nível médio em muitas tarefas.

**Faz sentido self-hospedar quando:**
1. **Volume é alto:** O ponto de equilíbrio entre API e infra própria costuma ficar em 1-5M tokens/dia dependendo do modelo
2. **Dados são sensíveis:** Saúde, jurídico, financeiro — dados que não podem sair do ambiente
3. **Latência ultra-baixa:** GPU local pode ser mais rápida que API com overhead de rede
4. **Personalização profunda:** Precisa de fine-tuning frequente em dados proprietários

### Ollama para Desenvolvimento e Testes

```python
# Ollama expõe uma API compatível com OpenAI
# Basta trocar a base_url

from openai import OpenAI

# Cliente apontando para Ollama local
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama não requer key real
)

def testar_modelo_local(prompt: str, modelo: str = "llama3.1") -> str:
    """
    Testa um modelo local via Ollama.
    Instale com: curl -fsSL https://ollama.ai/install.sh | sh
    Baixe modelo: ollama pull llama3.1
    """
    response = ollama_client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# A mesma abstração funciona para OpenAI, Ollama, vLLM...
```

### vLLM para Serving em Produção

```bash
# Instalar e iniciar servidor vLLM
pip install vllm

# Servir modelo com API compatível OpenAI
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --port 8000
```

```python
# Conectar ao vLLM com o mesmo cliente OpenAI
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm",  # Pode exigir token dependendo da config
)

# API idêntica — zero mudança no código de negócio
response = vllm_client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Olá!"}],
)
```

### Comparativo de Custo: API vs. Self-Hosted

```python
def comparar_custo_api_vs_selfhosted(
    tokens_por_dia: int,
    preco_api_por_milhao: float = 0.60,  # gpt-4o-mini saída
    custo_gpu_hora: float = 2.50,        # A100 40GB em cloud
    tokens_por_segundo_gpu: float = 50,  # throughput típico
) -> dict:
    """
    Compara custo mensal de API vs. GPU própria/alugada.
    """
    dias_por_mes = 30

    # Custo API
    tokens_mes = tokens_por_dia * dias_por_mes
    custo_api = (tokens_mes / 1_000_000) * preco_api_por_milhao

    # Custo GPU (alugada 24/7)
    horas_gpu_mes = 24 * dias_por_mes
    custo_gpu = custo_gpu_hora * horas_gpu_mes

    # Throughput da GPU (tokens que ela consegue processar)
    tokens_dia_gpu = tokens_por_segundo_gpu * 3600 * 24  # tokens/dia com 100% uso
    utilizacao_gpu = tokens_por_dia / tokens_dia_gpu

    return {
        "tokens_por_dia": tokens_por_dia,
        "custo_api_mensal_usd": round(custo_api, 2),
        "custo_gpu_mensal_usd": round(custo_gpu, 2),
        "utilizacao_gpu": f"{utilizacao_gpu:.1%}",
        "recomendacao": "self-hosted" if custo_gpu < custo_api else "api",
        "economia_mensal": round(abs(custo_api - custo_gpu), 2),
    }

# Análise de break-even
print("=== ANÁLISE API vs. GPU PRÓPRIA ===\n")
for volume in [100_000, 1_000_000, 5_000_000, 20_000_000]:
    resultado = comparar_custo_api_vs_selfhosted(volume)
    print(f"Volume: {volume:>12,} tokens/dia")
    print(f"  API:  ${resultado['custo_api_mensal_usd']:>8,.2f}/mês")
    print(f"  GPU:  ${resultado['custo_gpu_mensal_usd']:>8,.2f}/mês")
    print(f"  → {resultado['recomendacao'].upper()}")
    print()
```

**Saída esperada:**
```
=== ANÁLISE API vs. GPU PRÓPRIA ===

Volume:       100,000 tokens/dia
  API:   $      1.80/mês
  GPU:   $   1,800.00/mês
  → API

Volume:     5,000,000 tokens/dia
  API:   $     90.00/mês
  GPU:   $   1,800.00/mês
  → API

Volume:    20,000,000 tokens/dia
  API:   $    360.00/mês
  GPU:   $   1,800.00/mês
  → API    ← ainda não compensa para tokens de saída baratos!
```

> **Insight:** Para tokens de saída baratos (gpt-4o-mini), a API raramente perde para GPU própria em volume. O self-hosting faz mais sentido para **modelos mais caros** (GPT-4o equivalente) ou quando há **requisitos de privacidade**.

---

## 📌 Resumo da Parte 02

| Conceito | Definição Prática |
|----------|------------------|
| Token | Unidade de custo e processamento dos LLMs; ~4 chars em inglês, ~3-4 em português |
| Tokens de saída | Custam 3-5x mais que tokens de entrada — controle com `max_tokens` |
| Janela de contexto | Limite total de tokens por chamada; usar mais não é sempre melhor |
| Lost in the middle | Degradação de performance para informações no meio do contexto |
| TTFT | Time to First Token — latência percebida pelo usuário; use streaming |
| tiktoken | Biblioteca para contar tokens antes de enviar ao modelo |
| RPM / TPM | Rate limits por requisições e tokens por minuto; planeje antes |
| Exponential backoff | Estratégia de retry com espera crescente após erros de rate limit |
| TokenBudgetManager | Padrão para rastrear custo e truncar contexto automaticamente |
| Ollama | Ferramenta para rodar LLMs open source localmente (dev/testes) |
| vLLM | Framework de serving open source para LLMs em produção |

---

## 🔗 Referências

- [tiktoken no PyPI](https://github.com/openai/tiktoken) — Tokenizador oficial da OpenAI
- [OpenAI Tokenizer (visual)](https://platform.openai.com/tokenizer) — Ferramenta visual para entender tokenização
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Pesquisa sobre degradação em contextos longos
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits) — Documentação de limites por tier
- [Ollama](https://ollama.ai) — LLMs locais para desenvolvimento
- [vLLM](https://docs.vllm.ai) — Serving de LLMs open source em produção
- [LiteLLM](https://github.com/BerriAI/litellm) — Abstração multi-provedor para unificar API calls
- [Anthropic Model Comparison](https://docs.anthropic.com/en/docs/about-claude/models) — Tabela de modelos Claude

---

⬅️ **Anterior:** [Parte 01 — IA Generativa no Desenho de Sistemas](parte-01-ia-generativa-desenho-sistemas.md)  
➡️ **Próximo:** [Parte 03 — Prompt Engineering](parte-03-engenharia-de-contexto-1.md)  
🏠 **Início:** [README](../README.md)
