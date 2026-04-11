# Parte 01 — IA Generativa no Desenho de Sistemas

> **Carga horária:** 2h  
> **Prática correspondente:** [Prática 01](../praticas/pratica-01-primeiros-passos-llm.md)

---

## 1.1 IA como Componente de Sistema

Antes de escrever uma linha de código com IA, você precisa fazer a pergunta certa: **esse problema realmente precisa de um LLM?**

A maioria dos sistemas de software que existem hoje funciona muito bem sem IA generativa. Adicionar um LLM a um sistema tem custos reais — em dinheiro, latência, complexidade e imprevisibilidade. O entusiasmo do mercado não é um motivo técnico suficiente.

### O Mindset de Engenharia

Um engenheiro experiente vê um LLM como vê qualquer outro componente: ele tem uma interface, um custo de operação, uma taxa de falha e um contrato de qualidade. A diferença é que o "contrato" de qualidade de um LLM é **probabilístico**, não determinístico.

- Um banco de dados SQL retorna sempre o mesmo resultado para a mesma query.
- Um LLM pode retornar respostas diferentes para o mesmo prompt.

Isso não é bug, é característica — mas muda completamente como você projeta o sistema ao redor dele.

### As 3 Perguntas Antes de Adicionar IA

Antes de decidir usar um LLM, responda honestamente:

**1. O problema é mal-definido ou fuzzy?**  
Se a entrada e a saída esperada são claras e discretas, você provavelmente não precisa de IA. Processar um CSV, fazer um cálculo, validar um CPF — isso é trabalho de código determinístico.

**2. O problema se beneficia de compreensão de linguagem ou geração?**  
Classificar sentimento em texto livre, resumir documentos, gerar descrições de produto, responder perguntas em linguagem natural — aqui o LLM brilha.

**3. A barra de qualidade é fuzzy?**  
Se "bom o suficiente" é aceitável e humanos discordariam sobre qual é a resposta "certa", um LLM pode ser adequado. Se a resposta tem que ser 100% correta (cálculo financeiro, decisão regulatória), não delegue ao LLM sem validação determinística.

### Quando NÃO Usar LLM

| Problema | Por quê não usar LLM | Solução correta |
|----------|---------------------|-----------------|
| Ordenar uma lista de produtos por preço | Determinístico, O(n log n) | `sorted()` |
| Calcular juros compostos | Matemática exata requerida | Função Python |
| Verificar se email é válido | Regex resolve em microsegundos | `re.match()` |
| Roteamento de requisições HTTP | Regra de negócio clara | `if/elif` |
| Buscar registro por ID no banco | Indexado, previsível | Query SQL |
| Validar formato de CEP | Padrão fixo | Regex |

> **Regra prática:** Se você consegue escrever um teste unitário com `assert resultado == esperado`, provavelmente não precisa de LLM. Se o teste seria `assert resultado_parece_razoavel(resultado)`, aí o LLM pode ajudar.

---

## 1.2 Fronteiras do Produto e Decisões de Arquitetura

Uma vez decidido que um LLM faz sentido, a próxima decisão é: **como ele se encaixa na arquitetura?**

### LLM-as-Component (Recomendado para começar)

O modelo é uma caixa dentro de um sistema maior. Código determinístico controla o fluxo, valida entradas, pós-processa saídas e decide o que fazer com o resultado.

```
[Entrada do Usuário]
        ↓
[Validação Determinística] ← rejeita inputs malformados
        ↓
[Construção do Prompt]     ← lógica determinística
        ↓
    [LLM API]              ← o modelo faz UMA coisa bem definida
        ↓
[Parse da Resposta]        ← extrai estrutura da saída
        ↓
[Validação da Saída]       ← verifica se faz sentido
        ↓
[Lógica de Negócio]        ← código determinístico decide o que fazer
        ↓
[Resposta ao Usuário]
```

Exemplo de uso: classificar o tom de uma avaliação de produto (positivo/negativo/neutro) e salvar no banco.

### LLM-as-Orchestrator (Use com cautela)

O modelo decide o que fazer a seguir — chama ferramentas, decide fluxos, delega subtarefas. É a base dos sistemas de agentes.

```
[Entrada do Usuário]
        ↓
    [LLM]  →→→  [Ferramenta A]
      ↓               ↓
    [LLM]  ←←← [Resultado A]
      ↓
    [LLM]  →→→  [Ferramenta B]
      ↓               ↓
    [LLM]  ←←← [Resultado B]
      ↓
[Resposta Final]
```

**Riscos do padrão orchestrator:**
- Loops infinitos (o modelo continua chamando ferramentas)
- Custos imprevisíveis (quantas chamadas serão feitas?)
- Difícil de debugar (o que o modelo estava "pensando"?)
- Latência alta e variável

**Recomendação:** Comece com LLM-as-component. Só adote orchestrator quando tiver clareza sobre os limites do sistema.

### Onde Colocar os Guardrails

Guardrails são verificações que protegem o sistema de saídas ruins do LLM:

```python
class LLMComponent:
    def classify_sentiment(self, text: str) -> str:
        # Guardrail de entrada
        if len(text) > 5000:
            raise ValueError("Texto muito longo para classificação")
        if not text.strip():
            raise ValueError("Texto vazio")

        # Chamada ao LLM
        response = self._call_llm(text)

        # Guardrail de saída
        valid_values = {"positivo", "negativo", "neutro"}
        if response.lower() not in valid_values:
            # Fallback determinístico
            return "neutro"

        return response.lower()
```

---

## 1.3 Requisitos Não Funcionais com IA

Sistemas com IA têm requisitos não funcionais diferentes de sistemas tradicionais. Você precisa pensar neles **antes** de começar a construir.

### Latência: Qual é o Orçamento?

**Latência é o requisito não funcional mais subestimado em sistemas com LLM.**

| Contexto de Uso | Latência Aceitável | Estratégia |
|-----------------|-------------------|------------|
| Chat interativo em tempo real | < 500ms para primeiro token | Streaming obrigatório |
| Autocomplete enquanto digita | < 200ms | Modelos menores, cache agressivo |
| Resposta a pergunta pontual | < 3s total | Modelo balanceado |
| Geração de relatório (batch) | < 60s | Modelo maior, qualidade priorizada |
| Pipeline noturno de dados | Horas são ok | Máxima qualidade, menor custo |

**TTFT (Time to First Token)** é o que o usuário percebe como "velocidade". Mesmo que a geração total leve 10s, um TTFT de 300ms faz o sistema parecer responsivo. Streaming é essencial para UX interativa.

### Custo por Chamada

Todo engenheiro que usa LLMs em produção aprende — frequentemente de forma dolorosa — que o custo escala com o uso. Calcule **antes** de lançar.

```python
def estimar_custo_por_chamada(
    tokens_entrada: int,
    tokens_saida: int,
    modelo: str = "gpt-4o-mini"
) -> float:
    """Estima custo em USD por chamada à API."""
    precos = {
        # preço por 1M tokens (entrada, saida) em USD
        "gpt-4o":         (5.00,  15.00),
        "gpt-4o-mini":    (0.15,   0.60),
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25,   1.25),
    }
    if modelo not in precos:
        raise ValueError(f"Modelo desconhecido: {modelo}")

    preco_entrada, preco_saida = precos[modelo]
    custo = (tokens_entrada / 1_000_000) * preco_entrada
    custo += (tokens_saida / 1_000_000) * preco_saida
    return custo

# Exemplo
custo = estimar_custo_por_chamada(500, 200, "gpt-4o-mini")
print(f"Custo por chamada: ${custo:.6f}")  # ~$0.000195
```

### Confiabilidade: E Quando a API Cair?

APIs de LLM têm SLAs típicos de 99.9% — o que significa até 8.7 horas de downtime por ano. Seu sistema precisa sobreviver a isso.

Perguntas que você deve responder no design:
- O que acontece se a chamada ao LLM falhar?
- O sistema pode funcionar em modo degradado (sem IA)?
- Você tem um fallback determinístico?
- Qual é o timeout máximo aceitável?

### Risco de Vendor Lock-in

| Nível de Acoplamento | Exemplo | Risco |
|---------------------|---------|-------|
| Alto | Usa features específicas da OpenAI | Difícil trocar de provedor |
| Médio | Usa API OpenAI-compatible com modelo fixo | Pode trocar modelo, não interface |
| Baixo | Abstrai via LiteLLM ou interface própria | Troca de provedor é configuração |

**Mitigação prática:** Use uma camada de abstração fina desde o início.

```python
# ❌ Acoplado ao SDK da OpenAI
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4o", ...)

# ✅ Abstraído — troca de provider é só config
import litellm
response = litellm.completion(model="gpt-4o", ...)
# Trocar para Claude: model="claude-3-5-sonnet-20241022"
# Trocar para local: model="ollama/llama3"
```

---

## 1.4 Decisão de Build vs. Buy vs. Open Source

Uma das primeiras decisões arquiteturais é: onde o modelo roda?

### API Comercial (OpenAI, Anthropic, Google)

**Vantagens:**
- Começa em minutos, não semanas
- Qualidade de ponta sem infraestrutura
- Updates automáticos do modelo
- Escalabilidade transparente

**Desvantagens:**
- Custo variável (escala com uso)
- Dados saem da sua infraestrutura
- Dependência de terceiro
- Rate limits podem surpreender em picos

**Quando usar:** Prototipagem, produtos com volumes moderados, quando a qualidade do modelo importa mais que o custo de infra, dados não sensíveis.

### Modelo Open Source Self-Hosted (Ollama, vLLM, Hugging Face)

**Vantagens:**
- Custo fixo de infraestrutura (previsível)
- Dados ficam no seu ambiente
- Controle total sobre versão do modelo
- Sem rate limits externos

**Desvantagens:**
- Custo fixo alto mesmo sem uso
- Qualidade geralmente inferior aos modelos comerciais topo de linha
- Responsabilidade de manutenção e updates
- Precisa de GPU para performance aceitável

**Quando usar:** Dados sensíveis (saúde, financeiro, jurídico), volumes muito altos onde API é inviável economicamente, requisitos de air-gap.

### Fine-tuning: Quando e Por Quê (Raramente)

Fine-tuning é frequentemente a resposta a pergunta errada. A maioria dos casos de uso que "precisam" de fine-tuning na verdade precisam de:
- Prompt engineering melhor
- Exemplos no contexto (few-shot)
- RAG com dados relevantes

| Situação | Fine-tuning? | Alternativa |
|----------|-------------|-------------|
| Modelo não segue formato específico | Talvez | Few-shot examples no prompt |
| Modelo não conhece seu domínio | Não | RAG com sua base de conhecimento |
| Modelo muito genérico no tom | Não | System prompt detalhado |
| Latência alta, prompt muito longo | Sim | Fine-tune absorve o contexto |
| Custo alto por prompt longo repetido | Sim | Fine-tune reduz tokens |
| Tarefa muito específica e rara | Sim | Fine-tune com exemplos curados |

### Matriz de Decisão

| Critério | API Comercial | Open Source | Fine-tuned |
|----------|--------------|-------------|------------|
| Velocidade para produção | ⭐⭐⭐ | ⭐ | ⭐ |
| Qualidade do modelo | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Custo em escala | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Privacidade dos dados | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Controle e auditoria | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Complexidade de manutenção | ⭐⭐⭐ | ⭐ | ⭐ |

---

## 1.5 Padrões de Integração na Prática

### Síncrono vs. Assíncrono

**Chamada síncrona:** adequada para interações em tempo real onde o usuário espera a resposta.

**Chamada assíncrona:** necessária quando você não quer bloquear o thread enquanto espera o LLM, ou quando processa múltiplas requisições em paralelo.

```python
import asyncio
import httpx
from typing import Optional

async def chamar_llm_com_fallback(
    prompt: str,
    timeout_segundos: float = 10.0,
    fallback_response: Optional[str] = None
) -> str:
    """
    Chama o LLM com timeout e fallback.
    Em produção, use o SDK oficial com async support.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await asyncio.wait_for(
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                    }
                ),
                timeout=timeout_segundos
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    except asyncio.TimeoutError:
        print(f"LLM timeout após {timeout_segundos}s")
        return fallback_response or "Serviço temporariamente indisponível."
    except httpx.HTTPStatusError as e:
        print(f"Erro HTTP: {e.response.status_code}")
        return fallback_response or "Erro ao processar sua solicitação."
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return fallback_response or "Erro ao processar sua solicitação."


async def processar_lote(textos: list[str]) -> list[str]:
    """Processa múltiplos textos em paralelo com semáforo de controle."""
    semaforo = asyncio.Semaphore(5)  # máximo 5 chamadas simultâneas

    async def processar_com_limite(texto: str) -> str:
        async with semaforo:
            return await chamar_llm_com_fallback(texto)

    tarefas = [processar_com_limite(texto) for texto in textos]
    return await asyncio.gather(*tarefas)
```

### Arquitetura com Fila para Batch Processing

Para processar grandes volumes de forma resiliente, use fila de mensagens:

```
[Produtor]
    ↓ enfileira jobs
[Fila (Redis/SQS/RabbitMQ)]
    ↓ consome job
[Worker LLM]  →  [LLM API]
    ↓ salva resultado
[Banco de Dados]
    ↓
[Notifica produtor / atualiza status]
```

```python
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

@dataclass
class LLMJob:
    job_id: str
    prompt: str
    status: JobStatus = JobStatus.PENDING
    result: Optional[str] = None
    tentativas: int = 0
    max_tentativas: int = 3

class LLMWorker:
    """
    Worker simples que consome jobs de uma fila e chama o LLM.
    Em produção, use Celery, RQ, ou similar.
    """
    def __init__(self, fila: list, db: dict):
        self.fila = fila
        self.db = db

    def processar_job(self, job: LLMJob) -> None:
        job.status = JobStatus.PROCESSING
        job.tentativas += 1

        try:
            # Chama o LLM (simplificado)
            resultado = self._chamar_llm(job.prompt)
            job.result = resultado
            job.status = JobStatus.DONE

        except Exception as e:
            print(f"Job {job.job_id} falhou (tentativa {job.tentativas}): {e}")
            if job.tentativas >= job.max_tentativas:
                job.status = JobStatus.FAILED
            else:
                # Reencaminha para a fila com backoff
                time.sleep(2 ** job.tentativas)
                self.fila.append(job)

        self.db[job.job_id] = asdict(job)
```

### Degradação Graciosa

Seu sistema deve funcionar — mesmo que de forma reduzida — quando o LLM está indisponível:

```python
class SistemaComIA:
    def classificar_avaliacao(self, texto: str) -> dict:
        try:
            # Tenta classificação com IA (rico em contexto)
            sentimento = self.llm_client.classificar(texto)
            confianca = "alta"
        except Exception:
            # Fallback: análise determinística simples
            sentimento = self._classificar_por_palavras_chave(texto)
            confianca = "baixa"

        return {"sentimento": sentimento, "confianca": confianca}

    def _classificar_por_palavras_chave(self, texto: str) -> str:
        texto_lower = texto.lower()
        positivos = {"ótimo", "excelente", "adorei", "perfeito", "recomendo"}
        negativos = {"péssimo", "horrível", "detestei", "ruim", "não recomendo"}

        score = sum(1 for p in positivos if p in texto_lower)
        score -= sum(1 for n in negativos if n in texto_lower)

        if score > 0:
            return "positivo"
        elif score < 0:
            return "negativo"
        return "neutro"
```

---

## 1.6 Estimativa de Custo Antes de Construir

**Nunca lance um produto com LLM sem fazer as contas de custo.** Histórias de horror de faturas de API são reais.

### Calculadora de Custo Mensal

```python
from dataclasses import dataclass

@dataclass
class PerfilDeUso:
    usuarios_ativos_dia: int
    chamadas_por_usuario_dia: float
    tokens_entrada_media: int
    tokens_saida_media: int
    dias_por_mes: int = 30

@dataclass
class ConfigModelo:
    nome: str
    preco_entrada_por_milhao: float   # USD
    preco_saida_por_milhao: float     # USD

MODELOS = {
    "gpt-4o": ConfigModelo(
        "gpt-4o", 5.00, 15.00
    ),
    "gpt-4o-mini": ConfigModelo(
        "gpt-4o-mini", 0.15, 0.60
    ),
    "claude-3-5-sonnet": ConfigModelo(
        "claude-3-5-sonnet", 3.00, 15.00
    ),
    "claude-3-haiku": ConfigModelo(
        "claude-3-haiku", 0.25, 1.25
    ),
}

def estimar_custo_mensal(
    perfil: PerfilDeUso,
    modelo: ConfigModelo,
    margem_seguranca: float = 1.3
) -> dict:
    """
    Estima custo mensal com margem de segurança.
    margem_seguranca=1.3 → 30% acima da média estimada.
    """
    chamadas_mes = (
        perfil.usuarios_ativos_dia
        * perfil.chamadas_por_usuario_dia
        * perfil.dias_por_mes
    )
    tokens_entrada_mes = chamadas_mes * perfil.tokens_entrada_media
    tokens_saida_mes = chamadas_mes * perfil.tokens_saida_media

    custo_entrada = (tokens_entrada_mes / 1_000_000) * modelo.preco_entrada_por_milhao
    custo_saida = (tokens_saida_mes / 1_000_000) * modelo.preco_saida_por_milhao
    custo_base = custo_entrada + custo_saida
    custo_com_margem = custo_base * margem_seguranca

    return {
        "modelo": modelo.nome,
        "chamadas_mes": int(chamadas_mes),
        "tokens_entrada_mes": int(tokens_entrada_mes),
        "tokens_saida_mes": int(tokens_saida_mes),
        "custo_base_usd": round(custo_base, 2),
        "custo_estimado_usd": round(custo_com_margem, 2),
        "custo_estimado_brl": round(custo_com_margem * 5.0, 2),  # taxa aproximada
    }


# Exemplo de uso
perfil = PerfilDeUso(
    usuarios_ativos_dia=500,
    chamadas_por_usuario_dia=3,
    tokens_entrada_media=800,
    tokens_saida_media=300,
)

print("=== COMPARATIVO DE CUSTOS MENSAIS ===\n")
for nome_modelo, config in MODELOS.items():
    resultado = estimar_custo_mensal(perfil, config)
    print(f"�� {resultado['modelo']}")
    print(f"   Chamadas/mês: {resultado['chamadas_mes']:,}")
    print(f"   Custo estimado: ${resultado['custo_estimado_usd']:,.2f} USD")
    print(f"   (~R${resultado['custo_estimado_brl']:,.2f} BRL)\n")
```

**Saída esperada:**
```
=== COMPARATIVO DE CUSTOS MENSAIS ===

📦 gpt-4o
   Chamadas/mês: 45,000
   Custo estimado: $322.88 USD
   (~R$1,614.38 BRL)

📦 gpt-4o-mini
   Chamadas/mês: 45,000
   Custo estimado: $12.07 USD
   (~R$60.34 BRL)
```

### Planejamento do Orçamento de Tokens

Antes de escrever o prompt, decida o orçamento:

```python
class OrcamentoDeTokens:
    def __init__(self, limite_contexto: int = 128_000):
        self.limite_contexto = limite_contexto

    def planejar(
        self,
        system_prompt_tokens: int,
        historico_conversa_tokens: int,
        documento_contexto_tokens: int,
        reserva_saida_tokens: int,
    ) -> dict:
        total_entrada = (
            system_prompt_tokens
            + historico_conversa_tokens
            + documento_contexto_tokens
        )
        disponivel_para_entrada = self.limite_contexto - reserva_saida_tokens
        sobra = disponivel_para_entrada - total_entrada

        return {
            "total_entrada": total_entrada,
            "reserva_saida": reserva_saida_tokens,
            "total_alocado": total_entrada + reserva_saida_tokens,
            "percentual_usado": round(
                (total_entrada + reserva_saida_tokens) / self.limite_contexto * 100, 1
            ),
            "tokens_livres": sobra,
            "cabe_no_contexto": sobra >= 0,
        }

# Exemplo
orcamento = OrcamentoDeTokens(limite_contexto=128_000)
plano = orcamento.planejar(
    system_prompt_tokens=500,
    historico_conversa_tokens=2_000,
    documento_contexto_tokens=20_000,
    reserva_saida_tokens=2_000,
)
print(plano)
# {'total_entrada': 22500, 'reserva_saida': 2000, ..., 'percentual_usado': 19.1}
```

---

## 1.7 Checklist Antes de Começar

Use este checklist antes de escrever a primeira linha de código de integração com LLM:

### ✅ Validação do Problema
- [ ] O problema realmente precisa de LLM? (passou pelas 3 perguntas da seção 1.1?)
- [ ] Existe uma solução determinística mais simples e confiável?
- [ ] O valor de negócio justifica a complexidade adicionada?

### ✅ Arquitetura
- [ ] Decidi o padrão: LLM-as-component ou LLM-as-orchestrator?
- [ ] Defini onde ficam os guardrails de entrada e saída?
- [ ] Existe um fallback se o LLM estiver indisponível?
- [ ] A interface com o LLM está abstraída (não acoplada a um SDK específico)?

### ✅ Requisitos Não Funcionais
- [ ] Qual é o orçamento de latência? (e implementei streaming se necessário?)
- [ ] Calculei o custo mensal estimado para o volume esperado?
- [ ] O produto pode operar em modo degradado sem IA?
- [ ] Avaliei o risco de vendor lock-in?

### ✅ Dados e Privacidade
- [ ] Os dados enviados ao LLM podem sair da empresa?
- [ ] Existe PII (dados pessoais) no prompt? Está anonimizado?
- [ ] Verifiquei os termos de uso do provedor para o meu caso de uso?

### ✅ Operação
- [ ] Tenho logs de todas as chamadas ao LLM (sem dados sensíveis)?
- [ ] Há alertas de custo configurados no painel do provedor?
- [ ] Existe um limite (hard cap) de gasto mensal?
- [ ] Tenho métricas de qualidade para monitorar degradação?

---

## 📌 Resumo da Parte 01

| Conceito | Definição Prática |
|----------|------------------|
| LLM-as-component | O modelo faz uma tarefa específica; código determinístico controla o resto |
| LLM-as-orchestrator | O modelo decide o fluxo; maior risco, maior flexibilidade |
| Guardrail | Verificação determinística que protege entradas e saídas do LLM |
| Latência budget | Limite de tempo aceitável para resposta, define estratégia (streaming, cache) |
| TTFT | Time to First Token — o que o usuário percebe como "velocidade" |
| Degradação graciosa | Sistema funciona (de forma reduzida) mesmo sem o LLM disponível |
| Vendor lock-in | Risco de dependência de um único provedor; mitigue com abstração |
| Fine-tuning | Ajuste do modelo em dados específicos; raramente é a primeira solução certa |
| Token budget | Planejamento de quantos tokens cada componente do prompt pode usar |

---

## 🔗 Referências

- [OpenAI API Pricing](https://openai.com/pricing) — Tabela de preços atualizada
- [Anthropic Pricing](https://www.anthropic.com/pricing) — Preços Claude
- [LiteLLM](https://github.com/BerriAI/litellm) — Abstração multi-provedor
- [Building LLM applications for production — Chip Huyen](https://huyenchip.com/2023/04/11/llm-engineering.html) — Leitura obrigatória
- [Ollama](https://ollama.ai) — Rodar modelos localmente
- [vLLM](https://github.com/vllm-project/vllm) — Serving de LLMs open source em produção

---

⬅️ **Anterior:** — (Início do curso)  
➡️ **Próximo:** [Parte 02 — LLMs e Consumo de Contexto](parte-02-llms-consumo-contexto.md)  
🏠 **Início:** [README](../README.md)
