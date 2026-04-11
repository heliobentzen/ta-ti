# Parte 06 — Agentes no Sistema

> **Carga horária:** 7h  
> **Prática correspondente:** [Prática 06](../praticas/pratica-06-agentes.md)

---

## 6.1 O que é um agente? (definição prática, não filosófica)

O termo "agente" virou buzzword. Vamos ser diretos: **agente = LLM + ferramentas + loop**.

O LLM decide o que fazer. As ferramentas fazem acontecer coisas no mundo (chamar APIs, ler arquivos, executar código). O loop é o que permite que o agente execute múltiplos passos até completar uma tarefa.

Sem o loop, você tem um assistente que responde uma vez e para. Com o loop, você tem um sistema que planeja, age, observa o resultado, re-planeja, e age novamente — até terminar ou falhar.

### O padrão ReAct: Reason → Act → Observe

ReAct (Yao et al., 2022) é o padrão mais influente em agentes práticos. O modelo alterna entre:

```
Thought: Preciso verificar o saldo da conta do cliente X antes de processar.
Action: verificar_saldo(cliente_id="X123")
Observation: {"saldo": 1500.00, "moeda": "BRL", "status": "ativo"}

Thought: O saldo é suficiente. Vou processar a transferência de R$ 200.
Action: processar_transferencia(origem="X123", destino="Y456", valor=200.00)
Observation: {"status": "sucesso", "id_transacao": "TRX-789"}

Thought: Transferência concluída. Vou informar o usuário.
Action: FINISH
```

Cada iteração desse loop consome tokens. Cada `Action` chama uma ferramenta real. Cada `Observation` é o retorno dessa ferramenta injetado de volta no contexto.

### Quando agentes são apropriados

| Cenário | Use agente? | Por quê |
|---------|-------------|---------|
| Resposta simples com RAG | ❌ Não | Uma chamada + retrieval resolve |
| Fluxo de N passos fixo e conhecido | ❌ Não | Use código Python normal |
| Tarefa com múltiplos passos onde o próximo depende do anterior | ✅ Sim | O agente navega a incerteza |
| Integração com múltiplas APIs em sequência variável | ✅ Sim | O LLM decide a ordem |
| Tarefas de pesquisa e sumarização de múltiplas fontes | ✅ Sim | O agente orquestra o processo |

**Regra anti-hype:** se você consegue escrever o fluxo em Python sem usar um LLM para decidir o próximo passo, não use agente. Agentes adicionam custo, latência e não-determinismo. Use quando a tarefa tem ramificações que você não consegue prever em código.

### Agentes em produção vs. demos

Em demos, agentes fazem maravilhas. Em produção, você descobre:

- **Custo**: 10-50 iterações × custo de token = caro.
- **Latência**: cada passo do loop é uma chamada de API (200–2000ms). Fluxos de 10 passos = 2–20 segundos.
- **Confiabilidade**: ferramentas falham. APIs ficam fora. O agente precisa lidar com isso.
- **Loops infinitos**: o agente pode entrar em ciclo sem perceber.
- **Alucinação de ferramentas**: o modelo pode inventar argumentos ou chamar ferramentas com parâmetros inválidos.

Sistemas de agentes em produção têm limites explícitos, circuit breakers, logging robusto e fallbacks.

---

## 6.2 Ferramentas como contratos

A ferramenta é a interface entre o LLM e o mundo. A definição da ferramenta é o contrato que você assina com o modelo.

### O que compõe uma boa definição de ferramenta

1. **Nome claro e descritivo**: `buscar_pedido` é melhor que `get_data`
2. **Descrição que orienta quando usar**: o modelo lê a descrição para decidir se deve chamar a ferramenta
3. **Parâmetros bem tipados**: nome, tipo, descrição, se é obrigatório
4. **Exemplos quando ambíguo**: se o parâmetro tem formato específico, documente

A descrição da ferramenta é lida pelo LLM toda vez que ele decide se vai usá-la. Uma descrição ruim = ferramenta usada na hora errada ou não usada quando deveria.

### Definindo ferramentas no formato OpenAI

```python
from typing import Any

# Formato de definição de ferramenta compatível com OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_pedido",
            "description": (
                "Busca informações detalhadas sobre um pedido de compra pelo ID. "
                "Use quando o usuário mencionar um número de pedido ou quiser "
                "verificar o status de uma compra. Retorna status, itens e previsão de entrega."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pedido_id": {
                        "type": "string",
                        "description": "ID do pedido no formato PED-XXXXXX (ex: PED-123456)"
                    },
                    "incluir_historico": {
                        "type": "boolean",
                        "description": "Se True, inclui o histórico de atualizações do pedido",
                        "default": False
                    }
                },
                "required": ["pedido_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_pedidos_cliente",
            "description": (
                "Lista todos os pedidos de um cliente específico, opcionalmente "
                "filtrados por período ou status. Use quando o usuário quiser ver "
                "seu histórico de compras ou encontrar um pedido sem saber o ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cliente_id": {
                        "type": "string",
                        "description": "ID único do cliente"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pendente", "em_transito", "entregue", "cancelado"],
                        "description": "Filtra pedidos por status"
                    },
                    "limite": {
                        "type": "integer",
                        "description": "Número máximo de pedidos a retornar (padrão: 10)",
                        "default": 10
                    }
                },
                "required": ["cliente_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_frete",
            "description": (
                "Calcula o valor e prazo de frete para um CEP de destino dado o peso "
                "e dimensões do pacote. Use antes de confirmar compras ou quando o "
                "usuário perguntar sobre custo de entrega."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cep_destino": {
                        "type": "string",
                        "description": "CEP de destino no formato XXXXX-XXX ou XXXXXXXX"
                    },
                    "peso_gramas": {
                        "type": "number",
                        "description": "Peso do pacote em gramas"
                    },
                    "modalidade": {
                        "type": "string",
                        "enum": ["pac", "sedex", "sedex_10"],
                        "description": "Modalidade de envio"
                    }
                },
                "required": ["cep_destino", "peso_gramas"]
            }
        }
    }
]
```

### Wrapper Pythônico com decorador

```python
import functools
import inspect
import json
from typing import Callable, get_type_hints

TOOL_REGISTRY: dict[str, dict] = {}
TOOL_FUNCTIONS: dict[str, Callable] = {}


def tool(description: str):
    """
    Decorador que registra uma função como ferramenta de agente.

    Uso:
        @tool("Busca o clima atual de uma cidade")
        def buscar_clima(cidade: str, unidade: str = "celsius") -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            python_type = hints.get(name, str)
            json_type = _python_type_to_json(python_type)

            param_info = {"type": json_type}

            # Extrai descrição do docstring se disponível
            if fn.__doc__:
                doc_lines = fn.__doc__.strip().split("\n")
                for line in doc_lines:
                    if f"{name}:" in line:
                        param_info["description"] = line.split(":", 1)[1].strip()
                        break

            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                param_info["default"] = param.default

            properties[name] = param_info

        tool_def = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        }

        TOOL_REGISTRY[fn.__name__] = tool_def
        TOOL_FUNCTIONS[fn.__name__] = fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper
    return decorator


def _python_type_to_json(python_type) -> str:
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(python_type, "string")


# Uso do decorador
@tool("Busca o preço atual de um produto pelo seu código SKU")
def buscar_preco(sku: str, incluir_historico: bool = False) -> dict:
    """
    sku: Código SKU do produto (ex: PROD-001)
    incluir_historico: Se True, retorna histórico de preços dos últimos 30 dias
    """
    # Implementação real consultaria banco de dados / API
    return {"sku": sku, "preco": 99.90, "disponivel": True}


@tool("Verifica disponibilidade em estoque de um produto")
def verificar_estoque(sku: str, quantidade: int = 1) -> dict:
    """
    sku: Código SKU do produto
    quantidade: Quantidade desejada para verificação
    """
    return {"sku": sku, "disponivel": True, "quantidade_em_estoque": 42}
```

---

## 6.3 Integração com APIs e sistemas legados

Na prática, a maioria das ferramentas de agentes são wrappers em torno de APIs REST internas ou externas. O agente não sabe (nem deve saber) sobre HTTP, autenticação, rate limits — ele chama a função e recebe o resultado.

### Wrapper robusto com tratamento de erros

```python
import functools
import logging
import time
from typing import Any, Callable, Optional
import httpx  # pip install httpx

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Erro que o agente consegue entender e reagir."""
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 1.5,
    retryable_exceptions: tuple = (httpx.TimeoutException, httpx.ConnectError),
):
    """Decorator de retry com backoff exponencial."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait = backoff_factor ** (attempt - 1)
                        logger.warning(
                            f"{fn.__name__} falhou (tentativa {attempt}/{max_attempts}). "
                            f"Aguardando {wait:.1f}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"{fn.__name__} falhou após {max_attempts} tentativas: {e}"
                        )
            raise ToolError(
                f"Serviço indisponível após {max_attempts} tentativas: {last_exception}",
                retryable=False,
            )
        return wrapper
    return decorator


class APIClient:
    """Cliente HTTP reutilizável com autenticação e tratamento de erros."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @with_retry(max_attempts=3)
    def get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params, headers=self._headers)
            return self._handle_response(response)

    @with_retry(max_attempts=3)
    def post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=body, headers=self._headers)
            return self._handle_response(response)

    def _handle_response(self, response: httpx.Response) -> dict:
        if response.status_code == 404:
            raise ToolError(f"Recurso não encontrado: {response.url}", retryable=False)
        if response.status_code == 401:
            raise ToolError("Não autorizado: verifique as credenciais da API", retryable=False)
        if response.status_code == 429:
            raise ToolError("Rate limit atingido. Tente novamente em alguns segundos.", retryable=True)
        if response.status_code >= 500:
            raise ToolError(
                f"Erro interno do servidor ({response.status_code}): {response.text[:200]}",
                retryable=True,
            )
        if not response.is_success:
            raise ToolError(
                f"Erro HTTP {response.status_code}: {response.text[:200]}",
                retryable=False,
            )
        try:
            return response.json()
        except Exception:
            return {"raw_response": response.text}


# Ferramentas que usam o APIClient
import os

_erp_client = APIClient(
    base_url=os.getenv("ERP_API_URL", "https://erp.internal.example.com"),
    api_key=os.getenv("ERP_API_KEY"),
)


def buscar_cliente(cliente_id: str) -> dict:
    """Busca dados de um cliente no ERP."""
    try:
        return _erp_client.get(f"/clientes/{cliente_id}")
    except ToolError as e:
        # Propagamos o erro de forma que o agente possa entender
        return {
            "erro": str(e),
            "retryable": e.retryable,
            "acao_sugerida": (
                "Tente novamente mais tarde" if e.retryable
                else "Verifique o ID do cliente"
            )
        }


def criar_ticket_suporte(
    cliente_id: str,
    assunto: str,
    descricao: str,
    prioridade: str = "media",
) -> dict:
    """Cria um ticket de suporte no sistema de helpdesk."""
    if prioridade not in ("baixa", "media", "alta", "critica"):
        return {"erro": f"Prioridade inválida: {prioridade}. Use: baixa, media, alta, critica"}
    try:
        return _erp_client.post("/tickets", {
            "cliente_id": cliente_id,
            "assunto": assunto,
            "descricao": descricao,
            "prioridade": prioridade,
        })
    except ToolError as e:
        return {"erro": str(e), "ticket_criado": False}
```

---

## 6.4 O loop ReAct em prática

Vamos implementar o loop de agente do zero. Sem frameworks. Isso é importante para entender o que os frameworks fazem por baixo dos panos.

```python
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    thought: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[Any]
    is_final: bool = False
    final_answer: Optional[str] = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    total_tokens: int
    success: bool


class ReActAgent:
    """
    Implementação minimalista do loop ReAct.
    Propositalmente sem abstrações para fins didáticos.
    """

    def __init__(
        self,
        tools: list[dict],           # definições de ferramentas (formato OpenAI)
        tool_functions: dict[str, Callable],  # mapeamento nome -> função
        model: str = "gpt-4o-mini",
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
    ):
        self.client = OpenAI()
        self.tools = tools
        self.tool_functions = tool_functions
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt or (
            "Você é um assistente que pode usar ferramentas para responder perguntas. "
            "Use as ferramentas quando necessário. Quando tiver a resposta completa, "
            "responda diretamente ao usuário sem chamar mais ferramentas."
        )
        self.total_tokens = 0

    def run(self, user_message: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        steps = []
        step_count = 0

        while step_count < self.max_steps:
            step_count += 1
            logger.info(f"Passo {step_count}/{self.max_steps}")

            # Chama o LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            self.total_tokens += response.usage.total_tokens
            message = response.choices[0].message

            # Modelo não quer chamar ferramenta — tem a resposta final
            if not message.tool_calls:
                step = AgentStep(
                    thought=message.content,
                    tool_name=None,
                    tool_args=None,
                    tool_result=None,
                    is_final=True,
                    final_answer=message.content,
                )
                steps.append(step)
                logger.info(f"Agente concluiu em {step_count} passos.")
                return AgentResult(
                    answer=message.content,
                    steps=steps,
                    total_tokens=self.total_tokens,
                    success=True,
                )

            # Adiciona a mensagem do assistente (com tool_calls) ao histórico
            messages.append(message)

            # Executa cada tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                    logger.warning(f"Args inválidos para {tool_name}: {tool_call.function.arguments}")

                logger.info(f"Chamando ferramenta: {tool_name}({tool_args})")

                # Executa a ferramenta
                tool_result = self._execute_tool(tool_name, tool_args)

                step = AgentStep(
                    thought=message.content,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result,
                )
                steps.append(step)

                # Adiciona o resultado da ferramenta ao histórico
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Limite de passos atingido
        logger.warning(f"Agente atingiu limite de {self.max_steps} passos.")
        return AgentResult(
            answer="Não foi possível completar a tarefa dentro do limite de passos.",
            steps=steps,
            total_tokens=self.total_tokens,
            success=False,
        )

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        fn = self.tool_functions.get(tool_name)
        if fn is None:
            logger.error(f"Ferramenta desconhecida: {tool_name}")
            return {"erro": f"Ferramenta '{tool_name}' não existe."}
        try:
            return fn(**tool_args)
        except TypeError as e:
            logger.error(f"Argumentos inválidos para {tool_name}: {e}")
            return {"erro": f"Argumentos inválidos: {e}"}
        except Exception as e:
            logger.error(f"Erro ao executar {tool_name}: {e}", exc_info=True)
            return {"erro": f"Erro ao executar ferramenta: {e}"}


# Uso
if __name__ == "__main__":
    import os

    def buscar_pedido_mock(pedido_id: str, incluir_historico: bool = False) -> dict:
        return {
            "pedido_id": pedido_id,
            "status": "em_transito",
            "previsao_entrega": "2024-03-15",
            "itens": [{"produto": "Notebook", "quantidade": 1, "preco": 3500.00}],
        }

    def verificar_estoque_mock(sku: str, quantidade: int = 1) -> dict:
        return {"sku": sku, "disponivel": True, "quantidade_em_estoque": 5}

    agent = ReActAgent(
        tools=TOOL_DEFINITIONS,
        tool_functions={
            "buscar_pedido": buscar_pedido_mock,
            "verificar_estoque": verificar_estoque_mock,
        },
        model="gpt-4o-mini",
        max_steps=5,
    )

    result = agent.run("Qual é o status do pedido PED-123456 e tem Notebook em estoque (SKU: NB-001)?")
    print(f"Resposta: {result.answer}")
    print(f"Passos: {len(result.steps)}")
    print(f"Tokens usados: {result.total_tokens}")
```

---

## 6.5 Padrões de orquestração

### Padrão 1: Sequencial

Cada ferramenta é chamada na ordem, o resultado de uma alimenta a próxima.

```python
def fluxo_sequencial(agent: ReActAgent, etapas: list[str]) -> list[str]:
    """
    Executa etapas em sequência, passando contexto acumulado.
    Útil para workflows lineares com dependências entre etapas.
    """
    resultados = []
    contexto_acumulado = ""

    for etapa in etapas:
        prompt = f"{contexto_acumulado}\n\nPróxima tarefa: {etapa}" if contexto_acumulado else etapa
        result = agent.run(prompt)
        resultados.append(result.answer)
        # Acumula contexto para a próxima etapa
        contexto_acumulado = f"Resultado anterior: {result.answer}"

    return resultados


# Exemplo: pipeline de processamento de lead
etapas_lead = [
    "Busque os dados do cliente ID CL-001",
    "Com base nos dados do cliente, verifique se ele tem pedidos em aberto",
    "Crie um resumo personalizado para o representante de vendas",
]
```

### Padrão 2: Paralelo (com threading)

```python
import concurrent.futures
from typing import NamedTuple


class TaskResult(NamedTuple):
    task_name: str
    result: AgentResult
    error: Optional[Exception]


def executar_paralelo(
    tasks: dict[str, str],
    agent_factory: Callable[[], ReActAgent],
    max_workers: int = 3,
) -> dict[str, TaskResult]:
    """
    Executa múltiplas tarefas independentes em paralelo.

    tasks: {"nome_da_tarefa": "prompt para o agente"}
    agent_factory: função que cria um novo agente (cada thread precisa do seu)

    Cuidado: se as ferramentas têm estado compartilhado (ex: banco de dados),
    garanta thread-safety.
    """
    results = {}

    def run_task(task_name: str, task_prompt: str) -> TaskResult:
        agent = agent_factory()  # agente isolado por task
        try:
            result = agent.run(task_prompt)
            return TaskResult(task_name=task_name, result=result, error=None)
        except Exception as e:
            logger.error(f"Tarefa '{task_name}' falhou: {e}")
            return TaskResult(task_name=task_name, result=None, error=e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_task, name, prompt): name
            for name, prompt in tasks.items()
        }
        for future in concurrent.futures.as_completed(futures):
            task_result = future.result()
            results[task_result.task_name] = task_result

    return results


# Exemplo: análise de múltiplos produtos simultaneamente
tarefas_analise = {
    "produto_A": "Analise o estoque e preço do SKU PROD-001",
    "produto_B": "Analise o estoque e preço do SKU PROD-002",
    "produto_C": "Analise o estoque e preço do SKU PROD-003",
}
```

### Padrão 3: Condicional

```python
def fluxo_condicional(
    agent: ReActAgent,
    tarefa_inicial: str,
    rotas: dict[str, str],
    campo_decisao: str = "acao",
) -> AgentResult:
    """
    Executa uma tarefa e, baseado no resultado, escolhe a próxima.

    rotas: {"valor_do_campo": "próximo_prompt"}
    """
    resultado_inicial = agent.run(tarefa_inicial)

    # Tenta extrair o campo de decisão da resposta (assumindo JSON)
    try:
        dados = json.loads(resultado_inicial.answer)
        decisao = dados.get(campo_decisao)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Resposta não é JSON, usando rota padrão")
        decisao = None

    proximo_prompt = rotas.get(decisao) or rotas.get("default")
    if not proximo_prompt:
        return resultado_inicial

    contexto = f"Contexto da etapa anterior: {resultado_inicial.answer}\n\n{proximo_prompt}"
    return agent.run(contexto)


# Exemplo: triagem de tickets
ROTAS_TRIAGEM = {
    "tecnico": "Escale para o time de engenharia e registre com prioridade alta",
    "financeiro": "Encaminhe para o departamento financeiro com os dados do pedido",
    "cancelamento": "Inicie o fluxo de cancelamento e ofereça voucher de compensação",
    "default": "Registre o ticket como 'outros' e notifique o supervisor",
}
```

---

## 6.6 Modos de falha e loops infinitos

### O problema dos loops infinitos

Um agente pode entrar em loop quando:
- A ferramenta retorna erro mas o agente tenta de novo indefinidamente
- O agente fica pedindo confirmações de si mesmo
- A tarefa é ambígua e o agente fica refinando indefinidamente

O `max_steps` no `ReActAgent` é o circuit breaker básico. Mas há outras proteções:

```python
import hashlib
from collections import Counter


class LoopDetector:
    """Detecta padrões repetitivos no comportamento do agente."""

    def __init__(self, window_size: int = 3, max_repetitions: int = 2):
        self.window_size = window_size
        self.max_repetitions = max_repetitions
        self.call_history: list[str] = []

    def record_call(self, tool_name: str, tool_args: dict) -> None:
        signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        self.call_history.append(signature)

    def is_looping(self) -> bool:
        if len(self.call_history) < self.window_size * 2:
            return False

        recent = self.call_history[-self.window_size * self.max_repetitions :]
        counter = Counter(recent)

        for call, count in counter.items():
            if count >= self.max_repetitions:
                return True

        return False

    def last_calls_summary(self) -> str:
        recent = self.call_history[-5:]
        return " → ".join(recent)


class SafeReActAgent(ReActAgent):
    """ReActAgent com detecção de loop e orçamento de custo."""

    def __init__(
        self,
        *args,
        max_cost_usd: float = 0.50,
        cost_per_1k_tokens: float = 0.002,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_cost_usd = max_cost_usd
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.loop_detector = LoopDetector()

    @property
    def estimated_cost_usd(self) -> float:
        return (self.total_tokens / 1000) * self.cost_per_1k_tokens

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        # Registra a chamada para detecção de loop
        self.loop_detector.record_call(tool_name, tool_args)

        if self.loop_detector.is_looping():
            logger.error(
                f"Loop detectado! Últimas chamadas: {self.loop_detector.last_calls_summary()}"
            )
            raise RuntimeError("Agente entrou em loop — execução interrompida.")

        if self.estimated_cost_usd >= self.max_cost_usd:
            logger.error(
                f"Orçamento esgotado: ${self.estimated_cost_usd:.4f} >= ${self.max_cost_usd}"
            )
            raise RuntimeError(f"Orçamento de custo esgotado (${self.max_cost_usd}).")

        return super()._execute_tool(tool_name, tool_args)
```

### Custos por iteração

```python
# Referência de custo (GPT-4o-mini, jan/2025)
# Input: $0.15/1M tokens | Output: $0.60/1M tokens

def estimate_agent_cost(
    avg_context_tokens: int,
    avg_output_tokens: int,
    num_steps: int,
    model: str = "gpt-4o-mini",
) -> dict:
    """Estima o custo total de uma execução de agente."""
    prices = {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-5-haiku": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    }

    price = prices.get(model, prices["gpt-4o-mini"])
    total_input = avg_context_tokens * num_steps
    total_output = avg_output_tokens * num_steps

    cost = total_input * price["input"] + total_output * price["output"]
    return {
        "model": model,
        "estimated_steps": num_steps,
        "total_tokens": total_input + total_output,
        "estimated_cost_usd": round(cost, 6),
        "estimated_cost_brl": round(cost * 5.1, 4),
    }


# Agente de 10 passos com contexto médio de 2000 tokens
print(estimate_agent_cost(2000, 200, 10, "gpt-4o-mini"))
# {'model': 'gpt-4o-mini', 'estimated_steps': 10, 'total_tokens': 22000,
#  'estimated_cost_usd': 0.00042, 'estimated_cost_brl': 0.0021}

print(estimate_agent_cost(2000, 200, 10, "gpt-4o"))
# {'model': 'gpt-4o', 'estimated_steps': 10, 'total_tokens': 22000,
#  'estimated_cost_usd': 0.013, 'estimated_cost_brl': 0.066}
```

---

## 6.7 Tratamento de erros e retry

### Estratégias de tratamento de falha de ferramenta

```python
from enum import Enum


class ToolFailureStrategy(Enum):
    ABORT = "abort"           # para tudo
    RETRY = "retry"           # tenta de novo
    SKIP = "skip"             # continua sem o resultado
    FALLBACK = "fallback"     # usa alternativa


class ResilientAgent(ReActAgent):
    """
    Agente com estratégias configuráveis de tratamento de falha.
    """

    def __init__(
        self,
        *args,
        failure_strategy: ToolFailureStrategy = ToolFailureStrategy.RETRY,
        tool_retry_limit: int = 2,
        fallback_tools: dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.failure_strategy = failure_strategy
        self.tool_retry_limit = tool_retry_limit
        self.fallback_tools = fallback_tools or {}
        self._tool_attempt_counts: dict[str, int] = {}

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        attempt_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        self._tool_attempt_counts[attempt_key] = (
            self._tool_attempt_counts.get(attempt_key, 0) + 1
        )

        result = super()._execute_tool(tool_name, tool_args)

        # Verifica se o resultado indica erro
        if isinstance(result, dict) and "erro" in result:
            attempts = self._tool_attempt_counts[attempt_key]

            if self.failure_strategy == ToolFailureStrategy.RETRY:
                if attempts <= self.tool_retry_limit:
                    logger.info(f"Retentando {tool_name} (tentativa {attempts})")
                    time.sleep(1.0)
                    return self._execute_tool(tool_name, tool_args)
                else:
                    return {
                        "erro": result["erro"],
                        "mensagem_para_agente": (
                            f"A ferramenta {tool_name} falhou após {attempts} tentativas. "
                            "Considere uma abordagem alternativa ou informe ao usuário."
                        )
                    }

            elif self.failure_strategy == ToolFailureStrategy.FALLBACK:
                fallback_name = self.fallback_tools.get(tool_name)
                if fallback_name and fallback_name in self.tool_functions:
                    logger.info(f"Usando fallback {fallback_name} para {tool_name}")
                    return super()._execute_tool(fallback_name, tool_args)

            elif self.failure_strategy == ToolFailureStrategy.SKIP:
                return {
                    "aviso": f"Ferramenta {tool_name} indisponível. Continuando sem esse dado."
                }

        return result
```

### Graceful degradation

```python
def build_agent_with_degradation(
    primary_tools: dict,
    fallback_message: str = "Alguns serviços estão temporariamente indisponíveis.",
) -> ReActAgent:
    """
    Cria um agente que degrada graciosamente quando ferramentas falham.
    Ferramentas que falham retornam mensagens úteis em vez de lançar exceções.
    """
    wrapped_tools = {}

    for name, fn in primary_tools.items():
        def make_wrapper(func, tool_name):
            @functools.wraps(func)
            def wrapper(**kwargs):
                try:
                    return func(**kwargs)
                except Exception as e:
                    logger.error(f"Ferramenta {tool_name} falhou: {e}")
                    return {
                        "disponivel": False,
                        "ferramenta": tool_name,
                        "erro": str(e),
                        "instrucao": f"{fallback_message} Informe o usuário e ofereça alternativas."
                    }
            return wrapper
        wrapped_tools[name] = make_wrapper(fn, name)

    return ReActAgent(
        tools=TOOL_DEFINITIONS,
        tool_functions=wrapped_tools,
    )
```

---

## 6.8 Frameworks: LangGraph e smolagents

### A questão central: framework ou from scratch?

| Critério | Framework | From Scratch |
|----------|-----------|-------------|
| **Velocidade inicial** | Alta | Baixa |
| **Controle** | Limitado pela abstração | Total |
| **Debugging** | Difícil (muita magia) | Fácil (você escreveu) |
| **Manutenção** | Depende do projeto sobreviver | Você controla |
| **Complexidade** | Esconde complexidade | Você expõe e entende |

**Quando usar framework:** prototipagem rápida, equipe sem experiência com agentes, fluxos complexos com muitos nós.  
**Quando fazer from scratch:** você entende o que está fazendo, precisa de controle total, o framework adiciona mais complexidade do que resolve.

### LangGraph: orquestração baseada em grafos

LangGraph representa o fluxo do agente como um grafo dirigido. Cada nó é uma função Python. Arestas definem o fluxo condicional.

```python
# pip install langgraph langchain-openai
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int
    max_steps: int


def should_continue(state: AgentState) -> str:
    """Decide se continua ou para."""
    messages = state["messages"]
    last_message = messages[-1]

    if state["step_count"] >= state["max_steps"]:
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


def call_model(state: AgentState) -> dict:
    """Nó que chama o LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOL_DEFINITIONS)
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def call_tools(state: AgentState) -> dict:
    """Nó que executa as ferramentas."""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        fn = TOOL_FUNCTIONS.get(tool_call["name"])
        if fn:
            result = fn(**tool_call["args"])
        else:
            result = {"erro": f"Ferramenta desconhecida: {tool_call['name']}"}

        tool_messages.append(ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=tool_call["id"],
        ))

    return {"messages": tool_messages}


def build_langgraph_agent() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Uso
graph = build_langgraph_agent()
result = graph.invoke({
    "messages": [HumanMessage(content="Qual o status do pedido PED-001?")],
    "step_count": 0,
    "max_steps": 5,
})
print(result["messages"][-1].content)
```

### smolagents: minimalismo e transparência

```python
# pip install smolagents
from smolagents import CodeAgent, tool, HfApiModel


@tool
def buscar_clima(cidade: str) -> str:
    """
    Busca o clima atual de uma cidade.

    Args:
        cidade: Nome da cidade (ex: "Recife", "São Paulo")

    Returns:
        String descrevendo o clima atual
    """
    # Implementação real usaria API de clima
    return f"Em {cidade}: 28°C, parcialmente nublado, 65% umidade"


@tool
def converter_temperatura(celsius: float, para: str) -> float:
    """
    Converte temperatura de Celsius para outra unidade.

    Args:
        celsius: Temperatura em graus Celsius
        para: Unidade de destino ('fahrenheit' ou 'kelvin')

    Returns:
        Temperatura convertida
    """
    if para == "fahrenheit":
        return celsius * 9/5 + 32
    elif para == "kelvin":
        return celsius + 273.15
    raise ValueError(f"Unidade desconhecida: {para}")


# smolagents usa Code Agent por padrão: o LLM gera código Python que é executado
agent = CodeAgent(
    tools=[buscar_clima, converter_temperatura],
    model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
)

result = agent.run("Qual o clima em Recife e quanto é essa temperatura em Fahrenheit?")
print(result)
```

### Comparação honesta dos frameworks

| Framework | Pontos fortes | Pontos fracos | Melhor para |
|-----------|--------------|---------------|-------------|
| **LangGraph** | Fluxos complexos, visualização do grafo, estado persistente | Verboso, curva de aprendizado alta, muito boilerplate | Fluxos com muitos nós e ramificações complexas |
| **smolagents** | Simples, explícito, código Python real | Menos flexível para fluxos complexos | Agentes simples e médios, prototipagem |
| **AutoGen** | Multi-agente nativo, conversação entre agentes | Complexo, difícil de debugar | Sistemas com múltiplos agentes colaborativos |
| **CrewAI** | Abstração de "papéis" intuitiva | Pouco controle de baixo nível | Simulação de equipes, casos de negócio |
| **From scratch** | Controle total, simples de debugar | Você implementa tudo | Produção onde você entende cada linha |

---

## 6.9 Sistemas multi-agente

### Quando um agente não é suficiente

Um agente único fica limitado por:
- **Contexto**: acumular muitas informações ao longo de muitos passos infla o contexto e degrada a qualidade
- **Especialização**: um agente generalista é pior do que especialistas em tarefas específicas
- **Custo de coordenação**: tarefas longas ficam caras com um agente só

### Padrão Orquestrador + Especialistas

```python
class OrchestratorAgent:
    """
    Agente que delega para especialistas.
    O orquestrador entende a intenção e rota para o agente correto.
    """

    def __init__(self, specialists: dict[str, ReActAgent]):
        self.specialists = specialists
        self.llm = OpenAI()

    def route(self, task: str) -> str:
        """Usa LLM para decidir qual especialista usar."""
        specialist_list = "\n".join(
            f"- {name}" for name in self.specialists.keys()
        )
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Dado o seguinte especialistas disponíveis:\n{specialist_list}\n\n"
                    f"Qual deve processar esta tarefa: '{task}'\n\n"
                    f"Responda APENAS com o nome exato do especialista."
                )
            }],
        )
        specialist_name = response.choices[0].message.content.strip()
        return specialist_name

    def run(self, task: str) -> AgentResult:
        specialist_name = self.route(task)
        specialist = self.specialists.get(specialist_name)

        if specialist is None:
            logger.warning(f"Especialista '{specialist_name}' não encontrado, usando fallback")
            # Usa o primeiro especialista como fallback
            specialist = next(iter(self.specialists.values()))

        logger.info(f"Tarefa roteada para: {specialist_name}")
        return specialist.run(task)


# Configuração
agente_pedidos = ReActAgent(
    tools=[TOOL_DEFINITIONS[0]],
    tool_functions={"buscar_pedido": buscar_pedido_mock},
    system_prompt="Você é especialista em pedidos e entregas.",
)

agente_estoque = ReActAgent(
    tools=[TOOL_DEFINITIONS[1]],
    tool_functions={"verificar_estoque": verificar_estoque_mock},
    system_prompt="Você é especialista em estoque e disponibilidade de produtos.",
)

orquestrador = OrchestratorAgent(
    specialists={
        "agente_pedidos": agente_pedidos,
        "agente_estoque": agente_estoque,
    }
)
```

### Comunicação entre agentes: passagem de contexto estruturado

```python
import dataclasses
from datetime import datetime


@dataclasses.dataclass
class AgentMessage:
    """Mensagem estruturada entre agentes."""
    sender: str
    recipient: str
    content: str
    metadata: dict = dataclasses.field(default_factory=dict)
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now().isoformat()
    )


class AgentBus:
    """Barramento simples de mensagens entre agentes."""

    def __init__(self):
        self._queue: list[AgentMessage] = []
        self._agents: dict[str, ReActAgent] = {}

    def register(self, name: str, agent: ReActAgent) -> None:
        self._agents[name] = agent

    def send(self, message: AgentMessage) -> None:
        self._queue.append(message)
        logger.info(f"[{message.sender}] → [{message.recipient}]: {message.content[:80]}")

    def process_next(self) -> Optional[AgentResult]:
        if not self._queue:
            return None
        message = self._queue.pop(0)
        agent = self._agents.get(message.recipient)
        if agent is None:
            logger.error(f"Agente '{message.recipient}' não registrado")
            return None
        return agent.run(message.content)
```

---

## 6.10 Testando agentes

### O desafio de testar sistemas não-determinísticos

Testar agentes é difícil por três razões:
1. **Não-determinismo**: o mesmo prompt pode gerar chamadas de ferramentas diferentes
2. **Efeitos colaterais**: ferramentas modificam estado real (banco de dados, APIs)
3. **Custo**: cada teste chama o LLM

### Mockando ferramentas para testes

```python
import unittest
from unittest.mock import MagicMock, patch


class MockToolResponse:
    """Resposta pré-configurada para ferramentas em testes."""

    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
        self.call_log: list[tuple] = []

    def __call__(self, **kwargs) -> Any:
        self.call_log.append(kwargs)
        key = json.dumps(kwargs, sort_keys=True)
        if key in self.responses:
            return self.responses[key]
        # Retorno genérico se não configurado
        return {"status": "ok", "dados": "mock_data"}


class TestReActAgent(unittest.TestCase):

    def setUp(self):
        self.mock_buscar_pedido = MockToolResponse({
            '{"pedido_id": "PED-001"}': {
                "pedido_id": "PED-001",
                "status": "entregue",
                "data_entrega": "2024-03-10",
            }
        })

        self.agent = ReActAgent(
            tools=TOOL_DEFINITIONS[:1],
            tool_functions={"buscar_pedido": self.mock_buscar_pedido},
            model="gpt-4o-mini",
            max_steps=5,
        )

    def test_agent_uses_tool_for_order_query(self):
        """Agente deve chamar buscar_pedido quando perguntado sobre um pedido."""
        # Este teste é semi-determinístico: o LLM pode variar
        # Mas a intenção (chamar a ferramenta) deve ser consistente
        result = self.agent.run("Qual o status do pedido PED-001?")

        self.assertTrue(result.success)
        # Verifica que a ferramenta foi chamada
        self.assertEqual(len(self.mock_buscar_pedido.call_log), 1)
        self.assertEqual(
            self.mock_buscar_pedido.call_log[0]["pedido_id"],
            "PED-001"
        )
        # Verifica que a resposta menciona o status
        self.assertIn("entregue", result.answer.lower())

    def test_agent_handles_tool_error_gracefully(self):
        """Agente deve lidar com erros de ferramenta sem travar."""
        mock_erro = MagicMock(return_value={"erro": "Pedido não encontrado"})

        agent = ReActAgent(
            tools=TOOL_DEFINITIONS[:1],
            tool_functions={"buscar_pedido": mock_erro},
            model="gpt-4o-mini",
            max_steps=3,
        )

        result = agent.run("Qual o status do pedido PED-999?")
        # O agente deve completar (não lançar exceção) e comunicar o erro
        self.assertIsNotNone(result.answer)


def create_deterministic_test_set() -> list[dict]:
    """
    Conjunto de testes determinístico: casos onde o comportamento correto
    é claro e verificável independente do não-determinismo do LLM.
    """
    return [
        {
            "name": "pedido_existente",
            "input": "Status do PED-001",
            "expected_tool_call": "buscar_pedido",
            "expected_tool_args": {"pedido_id": "PED-001"},
            "mock_response": {"status": "entregue"},
            "validate_answer": lambda ans: "entregue" in ans.lower(),
        },
        {
            "name": "pedido_inexistente",
            "input": "Status do pedido XYZ-999",
            "expected_tool_call": "buscar_pedido",
            "mock_response": {"erro": "Não encontrado"},
            "validate_answer": lambda ans: len(ans) > 0,
        },
    ]
```

---

## 6.11 Panorama do ecossistema atual

Uma avaliação honesta do que está disponível em janeiro de 2025:

| Framework | Maturidade | Comunidade | Estabilidade de API | Ideal para |
|-----------|-----------|-----------|--------------------|----|
| **LangChain/LangGraph** | Alta | Muito grande | Média (muda muito) | Fluxos complexos, muitas integrações |
| **smolagents (HuggingFace)** | Média | Crescendo | Alta | Agentes simples e médios |
| **AutoGen (Microsoft)** | Alta | Grande | Alta | Multi-agente, pesquisa |
| **CrewAI** | Média | Média | Média | Simulação de equipes com papéis |
| **Pydantic AI** | Baixa (novo) | Pequena | Baixa (nova) | Validação tipada de respostas |
| **From scratch** | — | — | — | Produção com controle total |

### LangChain/LangGraph

**Pontos fortes:** ecossistema enorme, integrações prontas com quase tudo, LangGraph para fluxos complexos.  
**Pontos fracos:** abstração excessiva, debugging difícil, API muda frequentemente entre versões, curva de aprendizado íngreme.  
**Veredicto:** se você precisar de integração rápida com banco vetorial + LLM + agente, LangChain entrega. Se precisar de controle em produção, você vai refatorar para algo mais simples.

### smolagents

**Pontos fortes:** código explícito, fácil de entender e modificar, Code Agent é diferencial (o LLM escreve código Python que é executado).  
**Pontos fracos:** menos integrações prontas, ecossistema menor.  
**Veredicto:** para 80% dos casos de agentes simples e médios, smolagents é a escolha mais sã. Especialmente se você valoriza código legível.

### AutoGen

**Pontos fortes:** multi-agente nativo, bom para sistemas onde agentes conversam entre si, pesquisa ativa da Microsoft.  
**Pontos fracos:** complexo para casos simples, conversação entre agentes pode ser difícil de controlar.  
**Veredicto:** use quando o sistema realmente precisa de múltiplos agentes com papéis distintos colaborando.

### CrewAI

**Pontos fortes:** abstração de "crew" e "papéis" é intuitiva para casos de negócio.  
**Pontos fracos:** abstração pode esconder problemas, difícil de customizar profundamente.  
**Veredicto:** bom para demos e prototipagem. Em produção, a abstração costuma ser insuficiente.

### A recomendação prática

```
Complexidade baixa (1-3 ferramentas, fluxo simples)?
  → From scratch ou smolagents

Complexidade média (múltiplas ferramentas, algum fluxo condicional)?
  → smolagents ou LangGraph

Fluxo altamente complexo com muitos nós e estado persistente?
  → LangGraph

Múltiplos agentes colaborando?
  → AutoGen ou LangGraph multi-agent

Produção crítica onde você precisa entender cada linha?
  → From scratch
```

---

## �� Resumo da Parte 06

| Conceito | Definição |
|----------|-----------|
| **Agente** | LLM + ferramentas + loop de execução |
| **ReAct** | Padrão Reason → Act → Observe para execução de agentes |
| **Tool definition** | Contrato JSON que descreve uma ferramenta para o LLM |
| **max_steps** | Circuit breaker básico para evitar loops infinitos |
| **Loop detector** | Detecção de padrões repetitivos no comportamento do agente |
| **Orquestrador** | Agente que delega tarefas para agentes especialistas |
| **Parallel execution** | Execução simultânea de tarefas independentes com ThreadPoolExecutor |
| **Graceful degradation** | Continuar funcionando (de forma limitada) quando ferramentas falham |
| **LangGraph** | Framework de orquestração baseado em grafos dirigidos |
| **smolagents** | Framework minimalista com Code Agent (LLM gera código Python) |
| **Mock tools** | Ferramentas simuladas para testes determinísticos de agentes |
| **Custo por iteração** | Cada passo do loop consome tokens; agentes de 10 passos podem ser caros |

## 🔗 Referências

- [ReAct: Synergizing Reasoning and Acting in Language Models (paper original)](https://arxiv.org/abs/2210.03629)
- [LangGraph documentação](https://langchain-ai.github.io/langgraph/)
- [smolagents documentação (HuggingFace)](https://huggingface.co/docs/smolagents)
- [AutoGen (Microsoft)](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/crewAIInc/crewAI)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Building effective agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents)

---

⬅️ **Anterior:** [Parte 05](./parte-05-conhecimento-externo-rag.md) | ➡️ **Próximo:** [Parte 07](./parte-07-observabilidade-regressao.md)  
🏠 **Início:** [README](../README.md)
