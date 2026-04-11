# Parte 07 — Observabilidade e Regressão de Comportamento

> **Carga horária:** 2h  
> **Prática correspondente:** [Prática 07](../praticas/pratica-07-observabilidade.md)

---

## 7.1 Por que observabilidade é diferente em sistemas com LLM

Em sistemas tradicionais, observabilidade significa: métricas (CPU, latência, taxa de erro), logs (o que aconteceu) e traces (onde o tempo foi gasto). Você sabe se algo está errado porque há um erro HTTP 500, uma exception no log, ou uma métrica fora do limite.

Em sistemas com LLM, você pode ter um sistema que:
- **Retorna HTTP 200** com uma resposta incorreta
- **Não lança exceções** mas alucinoou
- **Funciona perfeitamente** para 95% dos casos mas falha sutilmente nos 5% mais importantes
- **Degradou silenciosamente** depois que você mudou o system prompt

Isso é fundamentalmente diferente. Você não sabe que algo está errado a não ser que você registre e avalie o que o modelo está produzindo.

### Os três problemas específicos de LLMs

**1. Não-determinismo:** O mesmo prompt pode gerar respostas diferentes. Sem logging de cada interação, você não consegue reproduzir bugs. "O modelo disse X ontem" é inútil sem o log exato da chamada.

**2. Custo invisível:** Cada token custa dinheiro. Uma feature nova que dobrou o tamanho do contexto pode ter dobrado seu custo sem nenhum alarme. Sem monitoramento de tokens e custo, a surpresa chega na fatura.

**3. Degradação de comportamento:** Um prompt que funcionava bem pode degradar silenciosamente se o modelo mudar (atualizações de versão), se a distribuição de inputs mudar, ou se o system prompt for editado com boas intenções mas efeito colateral inesperado.

### O que observabilidade de LLM precisa capturar

```
Chamada de LLM tradicional:
  latência + status code = suficiente

Chamada de LLM em produção:
  latência + tokens (input/output) + custo + modelo + versão do prompt +
  temperatura + sistema (hash) + usuário + sessão + resposta completa +
  qualidade da resposta (se possível avaliar)
```

---

## 7.2 O que registrar (e o que não registrar)

### O que você DEVE registrar

| Campo | Por quê |
|-------|---------|
| `messages` (input completo) | Reproduzir o problema exato |
| `response` (output completo) | Avaliar qualidade depois |
| `model` | Comportamento varia por modelo |
| `prompt_version` | Rastrear qual versão do prompt gerou o resultado |
| `input_tokens` | Custo e limites de contexto |
| `output_tokens` | Custo |
| `latency_ms` | Performance e SLA |
| `timestamp` | Correlação temporal |
| `user_id` | Debug por usuário, análise de segmento |
| `session_id` | Rastrear conversações inteiras |
| `cost_usd` | Alertas de custo |

### O que você DEVE REGISTRAR COM CUIDADO

**PII (Informação Pessoal Identificável) em prompts:**

Na era da LGPD (Lei Geral de Proteção de Dados), o prompt pode conter nome, CPF, email, endereço de um usuário. Registrar isso em logs sem criptografia ou anonimização é um risco legal sério.

```python
import re


def sanitize_pii_from_log(text: str) -> str:
    """
    Remove/mascara PII óbvia antes de registrar.
    IMPORTANTE: isto é uma heurística, não uma solução completa.
    Considere uma solução de PII detection dedicada para produção.
    """
    # CPF: 000.000.000-00 ou 00000000000
    text = re.sub(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}", "[CPF_REDACTED]", text)

    # Email
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL_REDACTED]", text)

    # Telefone brasileiro: (11) 99999-9999 ou 11999999999
    text = re.sub(r"(?:\(?\d{2}\)?\s?)(?:9\d{4}[\s-]?\d{4}|\d{4}[\s-]?\d{4})", "[PHONE_REDACTED]", text)

    # Cartão de crédito (16 dígitos, possivelmente com espaços/hífens)
    text = re.sub(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", "[CARD_REDACTED]", text)

    return text
```

### Wrapper de LLM com structured logging

```python
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger("llm.calls")


@dataclass
class LLMCallLog:
    call_id: str
    timestamp: str
    model: str
    prompt_version: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    user_id: Optional[str]
    session_id: Optional[str]
    success: bool
    error: Optional[str] = None
    # Não registrar o conteúdo completo em ambientes com LGPD sem anonimização
    input_hash: Optional[str] = None  # hash do input para correlação sem expor conteúdo
    response_preview: Optional[str] = None  # primeiros 100 chars

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# Preços por modelo (por token)
MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-4o-2024-11-20": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "claude-3-5-sonnet-20241022": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-3-5-haiku-20241022": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o-mini"])
    return input_tokens * prices["input"] + output_tokens * prices["output"]


class ObservableLLMClient:
    """
    Wrapper em volta do cliente OpenAI com logging estruturado.
    Drop-in replacement para chamadas diretas ao OpenAI.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_version: Optional[str] = None,
        log_content: bool = False,  # False em produção com dados sensíveis
    ):
        self.client = OpenAI()
        self.model = model
        self.prompt_version = prompt_version
        self.log_content = log_content

    def complete(
        self,
        messages: list[dict],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, LLMCallLog]:
        call_id = str(uuid.uuid4())
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model, input_tokens, output_tokens)

            import hashlib
            input_str = json.dumps(messages, ensure_ascii=False)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]

            log_entry = LLMCallLog(
                call_id=call_id,
                timestamp=timestamp,
                model=self.model,
                prompt_version=self.prompt_version,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost, 8),
                user_id=user_id,
                session_id=session_id,
                success=True,
                input_hash=input_hash,
                response_preview=content[:100] if self.log_content else None,
            )

            logger.info(log_entry.to_json())
            return content, log_entry

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_entry = LLMCallLog(
                call_id=call_id,
                timestamp=timestamp,
                model=self.model,
                prompt_version=self.prompt_version,
                input_tokens=0,
                output_tokens=0,
                latency_ms=round(latency_ms, 2),
                cost_usd=0.0,
                user_id=user_id,
                session_id=session_id,
                success=False,
                error=str(e),
            )
            logger.error(log_entry.to_json())
            raise
```

---

## 7.3 Rastreamento de custos em produção

Custo de LLM em produção é traiçoeiro. Uma feature que aumenta o contexto médio em 500 tokens por chamada, com 10.000 chamadas/dia, pode adicionar R$ 100-500/mês sem que ninguém perceba — até a fatura chegar.

### Tracker de custo com alertas

```python
import threading
from collections import defaultdict
from datetime import datetime, date


class CostTracker:
    """
    Rastreia custos de LLM por usuário, feature e modelo.
    Thread-safe para uso em ambientes de produção.
    """

    def __init__(self, daily_budget_usd: float = 50.0, alert_threshold: float = 0.8):
        self._lock = threading.Lock()
        self._costs: list[dict] = []
        self.daily_budget_usd = daily_budget_usd
        self.alert_threshold = alert_threshold
        self._alert_sent_today = False

    def record(
        self,
        cost_usd: float,
        model: str,
        user_id: Optional[str] = None,
        feature: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._costs.append({
                "timestamp": datetime.now().isoformat(),
                "date": date.today().isoformat(),
                "cost_usd": cost_usd,
                "model": model,
                "user_id": user_id or "anonymous",
                "feature": feature or "unknown",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            self._check_budget_alert()

    def _check_budget_alert(self) -> None:
        today = date.today().isoformat()
        today_cost = sum(
            r["cost_usd"] for r in self._costs if r["date"] == today
        )

        if (
            not self._alert_sent_today
            and today_cost >= self.daily_budget_usd * self.alert_threshold
        ):
            self._alert_sent_today = True
            self._send_alert(today_cost)

        # Reset flag no novo dia
        if self._costs and self._costs[-1]["date"] != today:
            self._alert_sent_today = False

    def _send_alert(self, current_cost: float) -> None:
        """Em produção, envie para Slack, PagerDuty, etc."""
        percentage = (current_cost / self.daily_budget_usd) * 100
        logger.critical(
            f"ALERTA DE CUSTO: ${current_cost:.4f} ({percentage:.0f}% do orçamento diário "
            f"de ${self.daily_budget_usd})"
        )
        # Exemplo: requests.post(SLACK_WEBHOOK, json={"text": f"Alerta: ..."})

    def daily_summary(self) -> dict:
        today = date.today().isoformat()
        today_records = [r for r in self._costs if r["date"] == today]

        if not today_records:
            return {"date": today, "total_usd": 0, "by_model": {}, "by_feature": {}, "by_user": {}}

        by_model = defaultdict(float)
        by_feature = defaultdict(float)
        by_user = defaultdict(float)

        for r in today_records:
            by_model[r["model"]] += r["cost_usd"]
            by_feature[r["feature"]] += r["cost_usd"]
            by_user[r["user_id"]] += r["cost_usd"]

        total = sum(r["cost_usd"] for r in today_records)

        return {
            "date": today,
            "total_usd": round(total, 6),
            "total_brl": round(total * 5.1, 4),
            "budget_used_pct": round((total / self.daily_budget_usd) * 100, 1),
            "calls": len(today_records),
            "by_model": dict(by_model),
            "by_feature": dict(by_feature),
            "top_users_by_cost": sorted(
                by_user.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def monthly_projection(self) -> dict:
        """Projeta custo mensal com base nos últimos 7 dias."""
        import statistics
        from collections import Counter

        daily_costs = defaultdict(float)
        for r in self._costs:
            daily_costs[r["date"]] += r["cost_usd"]

        if not daily_costs:
            return {"projection_usd": 0}

        recent_days = sorted(daily_costs.keys())[-7:]
        recent_values = [daily_costs[d] for d in recent_days]

        avg_daily = statistics.mean(recent_values)
        projection = avg_daily * 30

        return {
            "avg_daily_usd": round(avg_daily, 4),
            "monthly_projection_usd": round(projection, 2),
            "monthly_projection_brl": round(projection * 5.1, 2),
            "based_on_days": len(recent_days),
        }


# Instância global (singleton por processo)
cost_tracker = CostTracker(daily_budget_usd=50.0)
```

---

## 7.4 Regressão de prompts: como detectar que você quebrou algo

### O problema

Você tem um chatbot funcionando bem. Alguém (talvez você mesmo) edita o system prompt para melhorar um caso de uso. O chatbot continua respondendo — nenhuma exceção, nenhum erro. Mas 3 dias depois, os usuários começam a reclamar. O comportamento degradou silenciosamente.

Este é o problema de regressão de prompt. É a versão LLM de "funcionava na minha máquina".

### Conjuntos de avaliação: o golden test set

A solução é um conjunto de casos de teste com entradas, comportamentos esperados e critérios de avaliação. Esse conjunto precisa ser mantido e executado a cada mudança de prompt.

```python
import json
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EvalCase:
    """Um caso de teste para avaliação de prompt."""
    name: str
    input_messages: list[dict]
    # Pode ser uma string exata, ou uma função que recebe a resposta e retorna bool
    expected_behavior: str
    # Critério de sucesso: substring na resposta, regex, ou função customizada
    success_criterion: Callable[[str], bool]
    tags: list[str] = None
    weight: float = 1.0  # casos mais críticos têm peso maior


def contains_all(*substrings: str) -> Callable[[str], bool]:
    """Verifica se a resposta contém todas as substrings (case-insensitive)."""
    def check(response: str) -> bool:
        response_lower = response.lower()
        return all(s.lower() in response_lower for s in substrings)
    return check


def does_not_contain(*substrings: str) -> Callable[[str], bool]:
    """Verifica se a resposta NÃO contém certas substrings."""
    def check(response: str) -> bool:
        response_lower = response.lower()
        return all(s.lower() not in response_lower for s in substrings)
    return check


def response_is_json() -> Callable[[str], bool]:
    """Verifica se a resposta é JSON válido."""
    def check(response: str) -> bool:
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    return check


# Golden test set para um chatbot de suporte de e-commerce
GOLDEN_TEST_SET = [
    EvalCase(
        name="status_pedido_simples",
        input_messages=[
            {"role": "user", "content": "Onde está meu pedido PED-123?"}
        ],
        expected_behavior="Deve solicitar o ID completo ou confirmar que vai buscar",
        success_criterion=contains_all("pedido"),
        tags=["pedidos", "happy_path"],
        weight=1.0,
    ),
    EvalCase(
        name="reclamacao_tom_adequado",
        input_messages=[
            {"role": "user", "content": "Que absurdo! Meu produto chegou quebrado!"}
        ],
        expected_behavior="Deve se desculpar, mostrar empatia, oferecer solução",
        success_criterion=contains_all("desculp"),
        tags=["reclamacao", "tom"],
        weight=2.0,  # mais crítico
    ),
    EvalCase(
        name="pergunta_fora_do_escopo",
        input_messages=[
            {"role": "user", "content": "Qual é a capital da França?"}
        ],
        expected_behavior="Deve declinar gentilmente e redirecionar para o suporte",
        success_criterion=does_not_contain("paris", "france", "capital"),
        tags=["escopo", "guardrails"],
        weight=1.5,
    ),
    EvalCase(
        name="sem_inventar_politicas",
        input_messages=[
            {"role": "user", "content": "Vocês têm política de devolução de 60 dias?"}
        ],
        expected_behavior="Não deve confirmar política que não está no sistema prompt",
        success_criterion=does_not_contain("sim, 60 dias", "60 days"),
        tags=["alucinacao", "guardrails"],
        weight=3.0,  # crítico
    ),
]
```

### Runner de avaliação de regressão

```python
import time
from dataclasses import dataclass


@dataclass
class EvalResult:
    case_name: str
    passed: bool
    response: str
    latency_ms: float
    cost_usd: float
    error: Optional[str] = None


@dataclass
class EvalSuiteResult:
    prompt_version: str
    timestamp: str
    results: list[EvalResult]
    total_cost_usd: float
    total_latency_ms: float

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def weighted_pass_rate(self, test_set: list[EvalCase] = None) -> float:
        """Taxa de aprovação ponderada por peso do caso."""
        if not self.results:
            return 0.0
        total_weight = sum(tc.weight for tc in (test_set or []))
        if total_weight == 0:
            return self.pass_rate
        passed_weight = sum(
            tc.weight
            for tc, result in zip(test_set or [], self.results)
            if result.passed
        )
        return passed_weight / total_weight

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        failed = [r.case_name for r in self.results if not r.passed]
        return (
            f"Prompt: {self.prompt_version}\n"
            f"Resultado: {passed}/{total} casos aprovados ({self.pass_rate:.0%})\n"
            f"Custo total: ${self.total_cost_usd:.6f}\n"
            f"Casos reprovados: {failed if failed else 'nenhum'}"
        )


def run_eval_suite(
    test_cases: list[EvalCase],
    system_prompt: str,
    prompt_version: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
) -> EvalSuiteResult:
    """Executa o conjunto de avaliação e retorna resultados estruturados."""
    client = ObservableLLMClient(model=model, prompt_version=prompt_version)
    results = []
    total_cost = 0.0
    total_latency = 0.0

    for case in test_cases:
        messages = [{"role": "system", "content": system_prompt}] + case.input_messages

        try:
            start = time.time()
            response, log = client.complete(messages, max_tokens=max_tokens)
            latency = (time.time() - start) * 1000

            passed = case.success_criterion(response)
            total_cost += log.cost_usd
            total_latency += latency

            results.append(EvalResult(
                case_name=case.name,
                passed=passed,
                response=response,
                latency_ms=round(latency, 2),
                cost_usd=log.cost_usd,
            ))

            status = "✅" if passed else "❌"
            print(f"  {status} {case.name}")
            if not passed:
                print(f"     Esperado: {case.expected_behavior}")
                print(f"     Resposta: {response[:150]}...")

        except Exception as e:
            results.append(EvalResult(
                case_name=case.name,
                passed=False,
                response="",
                latency_ms=0,
                cost_usd=0,
                error=str(e),
            ))
            print(f"  💥 {case.name} - ERRO: {e}")

    return EvalSuiteResult(
        prompt_version=prompt_version,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        results=results,
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
    )


# Integração com CI/CD
def regression_check(
    new_system_prompt: str,
    new_prompt_version: str,
    baseline_pass_rate: float = 0.90,
) -> bool:
    """
    Retorna True se o novo prompt passa nos critérios mínimos.
    Use em pipelines de CI/CD para bloquear deploys regressivos.
    """
    print(f"\n🧪 Executando avaliação de regressão para versão: {new_prompt_version}")

    result = run_eval_suite(
        test_cases=GOLDEN_TEST_SET,
        system_prompt=new_system_prompt,
        prompt_version=new_prompt_version,
    )

    print(f"\n{result.summary()}")

    if result.pass_rate < baseline_pass_rate:
        print(
            f"\n🚨 REGRESSÃO DETECTADA: "
            f"{result.pass_rate:.0%} < {baseline_pass_rate:.0%} mínimo"
        )
        return False

    print(f"\n✅ Avaliação aprovada: {result.pass_rate:.0%} >= {baseline_pass_rate:.0%}")
    return True
```

---

## 7.5 Ferramentas de observabilidade para LLMs

### Langfuse: a melhor opção open-source para times

Langfuse é uma plataforma de observabilidade para LLMs que você pode hospedar você mesmo. Oferece rastreamento de chamadas, avaliações, comparação de prompts e análise de custo.

```python
# pip install langfuse openai
from langfuse import Langfuse
from langfuse.openai import openai  # monkey-patches o cliente openai

# Configura via variáveis de ambiente:
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (para self-hosted)

langfuse = Langfuse()


def chat_com_trace(
    messages: list[dict],
    user_id: str,
    session_id: str,
    prompt_version: str = "v1",
    model: str = "gpt-4o-mini",
) -> str:
    """
    Chamada ao LLM com rastreamento automático no Langfuse.
    O monkey-patch do openai captura automaticamente inputs/outputs/tokens.
    """
    trace = langfuse.trace(
        name="chat_completion",
        user_id=user_id,
        session_id=session_id,
        metadata={
            "prompt_version": prompt_version,
            "model": model,
        },
    )

    # Com o monkey-patch, esta chamada é automaticamente rastreada
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        # Langfuse injeta o trace_id automaticamente
    )

    content = response.choices[0].message.content

    # Registra score de qualidade (pode ser avaliação humana ou automática)
    langfuse.score(
        trace_id=trace.id,
        name="resposta_util",
        value=1,  # 0 ou 1, ou valor contínuo
    )

    return content


def avaliar_com_langfuse(trace_id: str, score: float, comment: str = "") -> None:
    """Registra avaliação de qualidade para um trace existente."""
    langfuse.score(
        trace_id=trace_id,
        name="qualidade",
        value=score,
        comment=comment,
    )


# Prompt management com Langfuse
def get_prompt_from_langfuse(prompt_name: str) -> str:
    """
    Busca prompt versionado do Langfuse.
    Permite editar prompts em produção sem redeploy.
    """
    prompt = langfuse.get_prompt(prompt_name)
    return prompt.compile()  # substitui variáveis se houver
```

### MLflow: se você já está no ecossistema ML

```python
# pip install mlflow openai
import mlflow
import mlflow.openai


def setup_mlflow_tracking(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)


def run_with_mlflow(
    messages: list[dict],
    system_prompt: str,
    prompt_version: str,
    model: str = "gpt-4o-mini",
) -> str:
    with mlflow.start_run():
        # Loga parâmetros
        mlflow.log_params({
            "model": model,
            "prompt_version": prompt_version,
            "temperature": 0.7,
        })

        client = ObservableLLMClient(model=model, prompt_version=prompt_version)
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response, log = client.complete(full_messages)

        # Loga métricas
        mlflow.log_metrics({
            "input_tokens": log.input_tokens,
            "output_tokens": log.output_tokens,
            "latency_ms": log.latency_ms,
            "cost_usd": log.cost_usd,
        })

        return response
```

### Logging estruturado custom: às vezes a resposta certa

Para muitos times, ferramentas externas adicionam dependência desnecessária. Um sistema de logging bem estruturado enviando para sua stack de observabilidade existente (Elasticsearch, Grafana Loki, CloudWatch) pode ser suficiente.

```python
import logging
import json
import sys


def setup_structured_logging() -> logging.Logger:
    """
    Configura logging JSON estruturado.
    Compatible com Elasticsearch, Grafana Loki, Splunk, etc.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("llm.observability")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Se o message já é JSON (nosso LLMCallLog), deserializa
        try:
            parsed = json.loads(record.getMessage())
            log_data.update(parsed)
            log_data.pop("message", None)
        except (json.JSONDecodeError, TypeError):
            pass

        return json.dumps(log_data, ensure_ascii=False)
```

### Comparação das ferramentas

| Ferramenta | Tipo | Custo | Self-hosted | Melhor para |
|------------|------|-------|-------------|-------------|
| **Langfuse** | Open-source | Gratuito (self-hosted) | ✅ Sim | Times que querem controle total |
| **LangSmith** | SaaS (LangChain) | Pago | ❌ Não | Quem usa LangChain/LangGraph |
| **MLflow** | Open-source | Gratuito | ✅ Sim | Quem já usa MLflow para ML |
| **Weights & Biases** | SaaS | Pago | Parcial | Times com cultura ML forte |
| **Custom logging** | — | Seu infra | ✅ Sim | Stacks existentes (ELK, Grafana) |

---

## 7.6 Sinais de degradação

Degradação silenciosa é um dos maiores riscos em produção com LLM. Estes são os sinais a monitorar:

### Sinais e como detectar

| Sinal | Indicador | Como detectar |
|-------|-----------|---------------|
| **Latência crescente** | p95 latency aumentando ao longo do tempo | Gráfico de percentil de latência |
| **Taxa de erro aumentando** | Mais chamadas falhando | Taxa de erro por hora/dia |
| **Custo por chamada aumentando** | Tokens médios por chamada subindo | `avg(input_tokens + output_tokens)` por dia |
| **Qualidade caindo** | Avaliações automáticas caindo | Score médio de LLM-as-judge ao longo do tempo |
| **Retrieval degradando (RAG)** | Recall@K caindo | Métricas de retrieval com dataset de teste fixo |
| **Feedback negativo aumentando** | Thumbs down de usuários | Taxa de feedback negativo |

### Detecção de anomalias simples

```python
import statistics
from collections import deque
from typing import Optional


class MetricAnomalyDetector:
    """
    Detector de anomalias baseado em z-score para métricas de LLM.
    Simples, sem dependências externas.
    """

    def __init__(self, window_size: int = 100, z_score_threshold: float = 3.0):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self._windows: dict[str, deque] = {}

    def record(self, metric_name: str, value: float) -> Optional[dict]:
        """
        Registra um valor e retorna alerta se for anomalia.
        Retorna None se normal, dict com detalhes se anomalia.
        """
        if metric_name not in self._windows:
            self._windows[metric_name] = deque(maxlen=self.window_size)

        window = self._windows[metric_name]

        # Precisa de dados suficientes para calcular z-score
        if len(window) < 10:
            window.append(value)
            return None

        mean = statistics.mean(window)
        stdev = statistics.stdev(window)

        window.append(value)

        if stdev == 0:
            return None

        z_score = abs((value - mean) / stdev)

        if z_score > self.z_score_threshold:
            direction = "alto" if value > mean else "baixo"
            return {
                "metric": metric_name,
                "value": value,
                "mean": round(mean, 4),
                "stdev": round(stdev, 4),
                "z_score": round(z_score, 2),
                "direction": direction,
                "alert": f"ANOMALIA: {metric_name}={value:.4f} está {direction} demais (z={z_score:.1f})",
            }

        return None


# Uso em produção
anomaly_detector = MetricAnomalyDetector(window_size=200, z_score_threshold=2.5)

def monitor_llm_call(log: LLMCallLog) -> None:
    """Monitora métricas de uma chamada e alerta anomalias."""
    metrics = {
        "latency_ms": log.latency_ms,
        "input_tokens": log.input_tokens,
        "output_tokens": log.output_tokens,
        "cost_usd": log.cost_usd,
    }

    for metric_name, value in metrics.items():
        alert = anomaly_detector.record(metric_name, value)
        if alert:
            logger.warning(json.dumps(alert))
            # Em produção: enviar para Slack, PagerDuty, etc.
```

### Dashboard de métricas simples

```python
from collections import defaultdict
from datetime import datetime, timedelta


class LLMMetricsDashboard:
    """Agregação de métricas para dashboard."""

    def __init__(self):
        self._records: list[LLMCallLog] = []

    def add(self, log: LLMCallLog) -> None:
        self._records.append(log)

    def summary_last_n_hours(self, hours: int = 24) -> dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            r for r in self._records
            if datetime.fromisoformat(r.timestamp.replace("Z", "")) > cutoff
        ]

        if not recent:
            return {"period_hours": hours, "calls": 0}

        latencies = [r.latency_ms for r in recent if r.success]
        input_tokens = [r.input_tokens for r in recent if r.success]
        costs = [r.cost_usd for r in recent]

        return {
            "period_hours": hours,
            "calls": len(recent),
            "success_rate": sum(1 for r in recent if r.success) / len(recent),
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "avg_input_tokens": round(statistics.mean(input_tokens), 0) if input_tokens else 0,
            "total_cost_usd": round(sum(costs), 6),
            "total_cost_brl": round(sum(costs) * 5.1, 4),
            "by_model": {
                model: sum(r.cost_usd for r in recent if r.model == model)
                for model in set(r.model for r in recent)
            },
        }
```

---

## 7.7 Experimentos e A/B testing de prompts

Você tem o prompt A e o prompt B. Qual é melhor? Não no seu julgamento — no comportamento real com usuários reais.

### Framework simples de A/B test

```python
import random
import hashlib
from dataclasses import dataclass


@dataclass
class Variant:
    name: str
    system_prompt: str
    weight: float = 0.5  # proporção do tráfego (0 a 1)


class PromptABTest:
    """
    Framework de A/B test para prompts.
    Usa hash do user_id para atribuição estável (o mesmo usuário sempre vê a mesma variante).
    """

    def __init__(self, variants: list[Variant], experiment_name: str):
        assert abs(sum(v.weight for v in variants) - 1.0) < 0.01, \
            "Pesos devem somar 1.0"
        self.variants = variants
        self.experiment_name = experiment_name
        self._results: dict[str, list[dict]] = {v.name: [] for v in variants}

    def assign_variant(self, user_id: str) -> Variant:
        """
        Atribuição estável por hash: o mesmo user_id sempre recebe a mesma variante.
        Garante experiência consistente para o usuário.
        """
        hash_val = int(hashlib.md5(f"{self.experiment_name}:{user_id}".encode()).hexdigest(), 16)
        rand_val = (hash_val % 10000) / 10000  # valor entre 0 e 1

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if rand_val < cumulative:
                return variant

        return self.variants[-1]

    def record_outcome(
        self,
        user_id: str,
        variant_name: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Registra outcome para análise estatística."""
        self._results[variant_name].append({
            "user_id": user_id,
            "metric": metric_name,
            "value": metric_value,
            "timestamp": datetime.now().isoformat(),
        })

    def analyze(self, metric_name: str) -> dict:
        """
        Análise estatística simples do experimento.
        Para significância estatística real, use scipy.stats.ttest_ind.
        """
        analysis = {}

        for variant in self.variants:
            records = [
                r["value"]
                for r in self._results[variant.name]
                if r["metric"] == metric_name
            ]
            if not records:
                analysis[variant.name] = {"n": 0, "mean": None}
                continue

            analysis[variant.name] = {
                "n": len(records),
                "mean": round(statistics.mean(records), 4),
                "stdev": round(statistics.stdev(records), 4) if len(records) > 1 else 0,
            }

        # Comparação entre variantes (se 2 variantes)
        if len(self.variants) == 2:
            v_a, v_b = self.variants[0].name, self.variants[1].name
            if analysis[v_a].get("mean") and analysis[v_b].get("mean"):
                diff = analysis[v_b]["mean"] - analysis[v_a]["mean"]
                analysis["comparison"] = {
                    "absolute_diff": round(diff, 4),
                    "relative_diff_pct": round(
                        (diff / analysis[v_a]["mean"]) * 100, 1
                    ) if analysis[v_a]["mean"] != 0 else None,
                    "winner": v_b if diff > 0 else v_a if diff < 0 else "empate",
                    "warning": "Execute teste estatístico formal antes de concluir.",
                }

        return analysis


# LLM-as-Judge para avaliação automatizada
def llm_as_judge(
    question: str,
    response: str,
    criteria: str,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    Usa um LLM mais forte para avaliar a qualidade de respostas.
    Útil para escalar avaliação sem anotação humana.

    CUIDADO: LLM-as-judge tem vieses conhecidos:
    - Preferência por respostas longas
    - Preferência pelo mesmo modelo (se for o mesmo)
    - Sensível ao formato do prompt de avaliação
    """
    client = OpenAI()

    judge_prompt = f"""Você é um avaliador especializado. Avalie a resposta abaixo.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA A AVALIAR:
{response}

CRITÉRIO DE AVALIAÇÃO:
{criteria}

Forneça:
1. Uma pontuação de 0 a 10 (10 = perfeito)
2. Uma justificativa em 1-2 frases
3. Problemas identificados (se houver)

Responda em JSON:
{{"score": <número>, "justificativa": "<texto>", "problemas": ["<item>", ...]}}"""

    response_obj = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response_obj.choices[0].message.content)
        result["judge_model"] = judge_model
        return result
    except json.JSONDecodeError:
        return {"score": -1, "erro": "Juiz não retornou JSON válido"}


# Exemplo de pipeline de avaliação contínua
def continuous_evaluation_pipeline(
    test_cases: list[EvalCase],
    system_prompt_v1: str,
    system_prompt_v2: str,
    judge_criteria: str = "A resposta é útil, precisa e adequada para suporte ao cliente?",
) -> dict:
    """Compara dois prompts usando LLM-as-judge."""
    client_v1 = ObservableLLMClient(prompt_version="v1")
    client_v2 = ObservableLLMClient(prompt_version="v2")

    scores = {"v1": [], "v2": []}

    for case in test_cases:
        for version, client, prompt in [
            ("v1", client_v1, system_prompt_v1),
            ("v2", client_v2, system_prompt_v2),
        ]:
            messages = [{"role": "system", "content": prompt}] + case.input_messages
            response, _ = client.complete(messages)

            question = case.input_messages[-1]["content"]
            judgment = llm_as_judge(question, response, judge_criteria)
            scores[version].append(judgment.get("score", 0))

    return {
        "v1_avg_score": round(statistics.mean(scores["v1"]), 2),
        "v2_avg_score": round(statistics.mean(scores["v2"]), 2),
        "improvement": round(
            statistics.mean(scores["v2"]) - statistics.mean(scores["v1"]), 2
        ),
        "recommendation": (
            "Deploy v2" if statistics.mean(scores["v2"]) > statistics.mean(scores["v1"])
            else "Manter v1"
        ),
    }
```

---

## 📌 Resumo da Parte 07

| Conceito | Definição |
|----------|-----------|
| **Observabilidade de LLM** | Logging de inputs, outputs, tokens, custo, latência para entender e debugar comportamento |
| **Non-determinismo** | O mesmo prompt pode gerar respostas diferentes; logging completo é essencial para reproduzir bugs |
| **PII em prompts** | Dados pessoais em prompts devem ser anonimizados ou criptografados (LGPD) |
| **Cost tracking** | Monitoramento contínuo de custo por chamada, usuário e feature |
| **Golden test set** | Conjunto de casos de teste com critérios de sucesso definidos para detectar regressões |
| **Regressão de prompt** | Degradação silenciosa de comportamento após mudança de prompt |
| **LLM-as-judge** | Uso de um LLM para avaliar automaticamente a qualidade de respostas de outro LLM |
| **A/B test de prompts** | Distribuição de tráfego entre variantes de prompt para comparar métricas reais |
| **Langfuse** | Plataforma open-source de observabilidade para LLMs; self-hostável |
| **Anomaly detection** | Detecção de métricas fora do padrão (latência, tokens, custo) com z-score |
| **Prompt versioning** | Rastreamento da versão do prompt usada em cada chamada |
| **Atribuição estável** | Hash do user_id garante que o mesmo usuário sempre vê a mesma variante no A/B test |

## 🔗 Referências

- [Langfuse — observabilidade open-source para LLMs](https://langfuse.com/)
- [MLflow LLM Tracking](https://mlflow.org/docs/latest/llms/index.html)
- [RAGAS — métricas de avaliação de RAG](https://github.com/explodinggradients/ragas)
- [LLM-as-judge: biases e limitações](https://arxiv.org/abs/2306.05685)
- [LGPD — Lei Geral de Proteção de Dados (Brasil)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [OpenTelemetry para LLMs](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Guia de A/B testing estatisticamente correto](https://www.exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)

---

⬅️ **Anterior:** [Parte 06](./parte-06-agentes-no-sistema.md) | ➡️ **Fim do curso!** 🎓  
🏠 **Início:** [README](../README.md)
