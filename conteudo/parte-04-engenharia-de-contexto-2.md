# Parte 04 — Engenharia de Contexto II

> **Carga horária:** 3h  
> **Prática correspondente:** [Prática 04](../praticas/pratica-04-engenharia-de-contexto-2.md)

---

## 4.1 Gerenciamento de Memória em Sistemas Conversacionais

O problema fundamental ao construir chatbots e assistentes com LLMs é simples: **a API é stateless**. Cada requisição à API é independente — o modelo não lembra de nenhuma conversa anterior. Toda memória que o sistema tem precisa ser explicitamente colocada no contexto de cada requisição.

Isso é diferente de como os humanos imaginam "conversar com uma IA". O usuário vê continuidade; o modelo vê um documento diferente a cada turn.

### Tipos de Memória

| Tipo | Onde fica | Custo | Capacidade | Acesso |
|------|-----------|-------|------------|--------|
| **In-context** | Janela de contexto do modelo | Por token | Limitada (ex: 128k tokens) | Imediato, perfeito recall |
| **Externa — cache** | Redis, Memcached | Baixo | Alta | Rápido (ms) |
| **Externa — vetorial** | ChromaDB, Pinecone, pgvector | Médio | Muito alta | Busca semântica |
| **Externa — banco** | PostgreSQL, MongoDB | Baixo | Muito alta | SQL/NoSQL query |

A decisão arquitetural central é: **quanto da memória fica no contexto vs. quanto fica externa e é recuperado sob demanda**.

### O Tradeoff Fundamental

Mais histórico no contexto = melhor compreensão da conversa, mas:
- Mais tokens de entrada = maior custo por requisição
- Latência aumenta (mais tokens para processar)
- Aproximação do limite de contexto
- Custo pode crescer indefinidamente em conversas longas

### Gerenciador Simples de Histórico com Budget de Tokens

```python
from openai import OpenAI
import tiktoken
from dataclasses import dataclass, field

client = OpenAI()

@dataclass
class ChatHistoryManager:
    """
    Gerencia histórico de conversa com controle de budget de tokens.
    Descarta mensagens mais antigas quando o limite é atingido.
    """
    modelo: str = "gpt-4o"
    max_tokens_historico: int = 4000  # budget para histórico
    mensagens: list[dict] = field(default_factory=list)

    def _contar_tokens(self, texto: str) -> int:
        enc = tiktoken.encoding_for_model(self.modelo)
        return len(enc.encode(texto))

    def _tokens_totais(self) -> int:
        return sum(
            self._contar_tokens(m["content"])
            for m in self.mensagens
        )

    def adicionar_mensagem(self, role: str, content: str):
        """Adiciona mensagem e trunca histórico se necessário."""
        self.mensagens.append({"role": role, "content": content})
        self._truncar_se_necessario()

    def _truncar_se_necessario(self):
        """Remove mensagens mais antigas até caber no budget."""
        while self._tokens_totais() > self.max_tokens_historico and len(self.mensagens) > 1:
            # Remove a mensagem mais antiga (preserva pelo menos a mais recente)
            self.mensagens.pop(0)

    def obter_mensagens(self, system_prompt: str) -> list[dict]:
        """Retorna mensagens formatadas para a API, com system prompt."""
        return [{"role": "system", "content": system_prompt}] + self.mensagens

    def chat(self, user_message: str, system_prompt: str) -> str:
        """Envia mensagem e retorna resposta, gerenciando o histórico."""
        self.adicionar_mensagem("user", user_message)

        response = client.chat.completions.create(
            model=self.modelo,
            messages=self.obter_mensagens(system_prompt)
        )
        assistant_message = response.choices[0].message.content
        self.adicionar_mensagem("assistant", assistant_message)
        return assistant_message

    def estatisticas(self) -> dict:
        return {
            "total_mensagens": len(self.mensagens),
            "tokens_em_uso": self._tokens_totais(),
            "budget_total": self.max_tokens_historico,
            "percentual_uso": f"{self._tokens_totais() / self.max_tokens_historico:.1%}"
        }

# Uso
manager = ChatHistoryManager(max_tokens_historico=3000)
SYSTEM = "Você é um assistente de programação Python. Seja direto e objetivo."

resposta1 = manager.chat("O que é uma list comprehension em Python?", SYSTEM)
print(f"Bot: {resposta1[:100]}...")

resposta2 = manager.chat("Me dê um exemplo com condição if", SYSTEM)
print(f"Bot: {resposta2[:100]}...")

print(f"\nEstatísticas: {manager.estatisticas()}")
```

---

## 4.2 Estratégias de Memória Baseada em Resumo

A estratégia de truncação simples (descartar o mais antigo) funciona para conversas curtas, mas falha em conversas longas onde informações antigas são relevantes. A solução é **rolling summary**: resumir turnos antigos e manter apenas o resumo no contexto.

### O Padrão Rolling Summary

```
Antes:
[turn 1: usuário explica seu problema de negócio complexo]
[turn 2: assistente esclarece dúvidas]
[turn 3: usuário fornece dados adicionais]
[turn 4: assistente faz análise]
...
[turn 15: pergunta atual]

Depois (com rolling summary):
[RESUMO: usuário tem e-commerce de moda, 50k usuários/mês, quer reduzir abandono
de carrinho de 73% para <50%. Já tentou emails de recuperação sem sucesso.
Decidiu implementar notificações push. Dados: conversão atual 1.2%]
[turn 13: detalhe técnico relevante recente]
[turn 14: resposta do assistente]
[turn 15: pergunta atual]
```

O resumo preserva o contexto semântico sem o custo total dos turnos originais.

### Quando Triggerar Resumização

```python
from openai import OpenAI
import tiktoken
from dataclasses import dataclass, field

client = OpenAI()

@dataclass
class SummaryMemoryManager:
    """
    Gerencia memória com estratégia de resumo rolante.
    Quando o histórico excede o threshold, os turnos mais antigos
    são resumidos e o resumo substitui as mensagens originais.
    """
    modelo: str = "gpt-4o"
    modelo_resumo: str = "gpt-4o-mini"  # modelo mais barato para resumir
    max_tokens_antes_resumo: int = 3000
    turnos_a_preservar: int = 4  # últimos N turnos mantidos verbatim
    resumo_atual: str = ""
    mensagens_recentes: list[dict] = field(default_factory=list)

    def _contar_tokens(self, texto: str) -> int:
        enc = tiktoken.encoding_for_model(self.modelo)
        return len(enc.encode(texto))

    def _tokens_recentes(self) -> int:
        return sum(self._contar_tokens(m["content"]) for m in self.mensagens_recentes)

    def _resumir_mensagens(self, mensagens: list[dict]) -> str:
        """Usa LLM para criar resumo compacto de mensagens antigas."""
        historico_texto = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in mensagens
        )
        prompt = (
            "Crie um resumo conciso do seguinte trecho de conversa. "
            "Preserve: decisões tomadas, fatos importantes, preferências do usuário, "
            "e contexto necessário para continuar a conversa.\n\n"
            f"Conversa:\n{historico_texto}\n\n"
            "Resumo (máximo 200 palavras):"
        )
        response = client.chat.completions.create(
            model=self.modelo_resumo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        return response.choices[0].message.content

    def adicionar_mensagem(self, role: str, content: str):
        """Adiciona mensagem e triggera resumo se necessário."""
        self.mensagens_recentes.append({"role": role, "content": content})
        self._verificar_e_resumir()

    def _verificar_e_resumir(self):
        """Verifica se deve resumir e executa se necessário."""
        tokens_em_uso = self._tokens_recentes()
        if (tokens_em_uso > self.max_tokens_antes_resumo and
                len(self.mensagens_recentes) > self.turnos_a_preservar * 2):

            # Separa mensagens a resumir das que ficam verbatim
            ponto_corte = len(self.mensagens_recentes) - (self.turnos_a_preservar * 2)
            mensagens_antigas = self.mensagens_recentes[:ponto_corte]
            self.mensagens_recentes = self.mensagens_recentes[ponto_corte:]

            # Cria novo resumo combinando resumo anterior com mensagens antigas
            conteudo_para_resumir = (
                (f"[Resumo anterior]\n{self.resumo_atual}\n\n[Continuação]\n"
                 if self.resumo_atual else "")
            )
            self.resumo_atual = self._resumir_mensagens(mensagens_antigas)
            print(f"[MemoryManager] Resumo gerado: {len(mensagens_antigas)} mensagens condensadas")

    def obter_mensagens_para_api(self, system_prompt: str) -> list[dict]:
        """Monta as mensagens para a API, incluindo resumo se existir."""
        mensagens = [{"role": "system", "content": system_prompt}]

        if self.resumo_atual:
            # Injeta resumo como contexto no system ou como mensagem de sistema
            mensagens.append({
                "role": "system",
                "content": f"[Contexto da conversa anterior]\n{self.resumo_atual}"
            })

        mensagens.extend(self.mensagens_recentes)
        return mensagens

    def chat(self, user_message: str, system_prompt: str) -> str:
        self.adicionar_mensagem("user", user_message)
        response = client.chat.completions.create(
            model=self.modelo,
            messages=self.obter_mensagens_para_api(system_prompt)
        )
        assistant_message = response.choices[0].message.content
        self.adicionar_mensagem("assistant", assistant_message)
        return assistant_message
```

### Preservando Entidades Críticas Across Summaries

Resumos automáticos podem perder informações específicas importantes. A solução é extrair entidades-chave explicitamente:

```python
def extrair_entidades_criticas(historico: list[dict]) -> dict:
    """
    Extrai entidades críticas que DEVEM ser preservadas no contexto.
    Ex: nome do usuário, decisões tomadas, preferências declaradas.
    """
    historico_texto = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in historico
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extraia as seguintes informações da conversa no formato JSON:\n"
                    "- nome_usuario: nome do usuário se mencionado\n"
                    "- decisoes: lista de decisões tomadas\n"
                    "- preferencias: preferências declaradas\n"
                    "- fatos_chave: informações factuais importantes\n"
                    "Se uma informação não existir, use null."
                )
            },
            {"role": "user", "content": historico_texto}
        ],
        temperature=0
    )
    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {}
```

---

## 4.3 Arquiteturas Stateless vs. Stateful

A escolha entre arquitetura stateless e stateful para sistemas conversacionais tem implicações profundas em custo, complexidade e escalabilidade.

### Stateless: Cada Requisição é Autocontida

Em uma arquitetura stateless, o cliente envia todo o histórico relevante em cada requisição. O servidor não mantém estado entre chamadas.

```
Cliente → [system + histórico completo + nova mensagem] → API
Cliente ← [resposta]                                   ← API
```

**Vantagens:**
- Simplicidade: sem gerenciamento de sessão no servidor
- Escalabilidade horizontal perfeita: qualquer instância pode atender qualquer requisição
- Cache-friendly: com prefixo estável, aproveita KV cache dos modelos
- Resiliente: sem estado para perder em falhas

**Desvantagens:**
- Payload cresce com cada turn (custo de transferência)
- Cliente precisa manter e enviar o histórico
- Conversas muito longas ficam caras rapidamente

### Stateful: Servidor Mantém o Estado da Sessão

Em arquitetura stateful, o servidor armazena o histórico de cada sessão. O cliente envia apenas a nova mensagem.

```
Cliente → [session_id + nova mensagem] → Servidor
Servidor → recupera histórico do Redis/DB
Servidor → [system + histórico + nova mensagem] → API
Servidor ← [resposta]                          ← API
Servidor → salva resposta na sessão
Cliente ← [resposta]                          ← Servidor
```

**Vantagens:**
- Payload pequeno por requisição
- Controle total sobre o contexto no servidor
- Pode aplicar políticas complexas (filtros, compressão, logging)

**Desvantagens:**
- Complexidade de gerenciamento de sessão
- Session affinity ou storage centralizado necessário
- Single point of failure se o storage cair

### Quando Usar Cada Arquitetura

| Critério | Stateless | Stateful |
|----------|-----------|---------|
| Conversas curtas (<10 turns) | Prefira | Overhead desnecessário |
| Conversas longas e complexas | Custo cresce | Prefira |
| Múltiplos clientes (mobile, web) | Simples | Necessita storage centralizado |
| Compliance e auditoria | Difícil de centralizar | Ideal |
| Escalabilidade horizontal | Trivial | Requer Redis/DB distribuído |
| Latência por requisição | Maior (payload) | Menor (só nova mensagem) |

### Implementação Stateful com Redis

```python
import json
import time
from openai import OpenAI

# Simula interface Redis para exemplo (use redis-py em produção)
class RedisSimulado:
    def __init__(self):
        self._store = {}

    def get(self, key: str) -> str | None:
        item = self._store.get(key)
        if item and item["expires_at"] > time.time():
            return item["value"]
        return None

    def setex(self, key: str, seconds: int, value: str):
        self._store[key] = {
            "value": value,
            "expires_at": time.time() + seconds
        }

    def delete(self, key: str):
        self._store.pop(key, None)

client = OpenAI()
redis = RedisSimulado()

SESSION_TTL = 3600  # 1 hora de inatividade antes de expirar

def chat_stateful(session_id: str, user_message: str, system_prompt: str) -> str:
    """
    Endpoint de chat stateful.
    Recupera histórico da sessão, processa, salva de volta.
    """
    # Recuperar histórico da sessão
    historico_json = redis.get(f"session:{session_id}")
    historico = json.loads(historico_json) if historico_json else []

    # Montar mensagens para a API
    mensagens = [{"role": "system", "content": system_prompt}]
    mensagens.extend(historico)
    mensagens.append({"role": "user", "content": user_message})

    # Chamar API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=mensagens
    )
    assistant_reply = response.choices[0].message.content

    # Atualizar histórico
    historico.append({"role": "user", "content": user_message})
    historico.append({"role": "assistant", "content": assistant_reply})

    # Salvar sessão com TTL renovado
    redis.setex(f"session:{session_id}", SESSION_TTL, json.dumps(historico))

    return assistant_reply

# Teste
SYSTEM = "Você é um assistente de suporte técnico."
sid = "user-123-session-abc"

r1 = chat_stateful(sid, "Meu Python não está instalado", SYSTEM)
r2 = chat_stateful(sid, "Estou no Windows 11", SYSTEM)  # modelo "lembra" do contexto
print(r1[:80])
print(r2[:80])
```

---

## 4.4 O Que Incluir no Contexto a Cada Turno

Nem tudo no histórico é igualmente relevante para o turn atual. A seleção inteligente de contexto pode reduzir custos significativamente sem perda de qualidade.

### Três Dimensões de Relevância

**1. Recência** — Turnos mais recentes são geralmente mais relevantes:

```python
def selecionar_por_recencia(historico: list[dict], max_turnos: int) -> list[dict]:
    """Mantém apenas os N turnos mais recentes."""
    # Preserva pares completos (usuário + assistente)
    pares = [(historico[i], historico[i+1])
             for i in range(0, len(historico)-1, 2)
             if i+1 < len(historico)]
    return [msg for par in pares[-max_turnos:] for msg in par]
```

**2. Relevância semântica** — Nem todo histórico é relevante para a pergunta atual:

```python
def selecionar_por_relevancia(
    historico: list[dict],
    pergunta_atual: str,
    top_k: int = 3
) -> list[dict]:
    """
    Seleciona os turnos mais semanticamente relevantes para a pergunta.
    Usa embeddings para medir similaridade.
    Requer: openai embeddings ou sentence-transformers
    """
    if not historico:
        return []

    # Gera embedding da pergunta atual
    resp_query = client.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta_atual
    )
    query_vec = resp_query.data[0].embedding

    # Gera embeddings dos turnos do usuário (proxy de relevância do par)
    turnos_usuario = [
        (i, m) for i, m in enumerate(historico)
        if m["role"] == "user"
    ]

    if not turnos_usuario:
        return historico

    resp_turnos = client.embeddings.create(
        model="text-embedding-3-small",
        input=[m["content"] for _, m in turnos_usuario]
    )

    # Calcula similaridade cosseno
    import math

    def cosseno(a: list[float], b: list[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

    similaridades = [
        (idx, cosseno(query_vec, emb.embedding))
        for (idx, _), emb in zip(turnos_usuario, resp_turnos.data)
    ]

    # Seleciona top-k mais relevantes
    top_indices = {
        idx for idx, _ in sorted(similaridades, key=lambda x: x[1], reverse=True)[:top_k]
    }

    # Inclui pares completos dos turnos selecionados
    resultado = []
    for i, msg in enumerate(historico):
        if i in top_indices or (i > 0 and (i-1) in top_indices):
            resultado.append(msg)

    return resultado
```

**3. Importância declarada** — Alguns fatos precisam ser sempre preservados:

```python
@dataclass
class ContextoConversacional:
    """
    Mantém contexto estruturado com camadas de importância.
    Fatos importantes são sempre incluídos, independente do budget.
    """
    fatos_permanentes: dict = field(default_factory=dict)  # sempre incluídos
    historico: list[dict] = field(default_factory=list)

    def definir_fato_permanente(self, chave: str, valor: str):
        """Registra fato que deve sempre estar no contexto."""
        self.fatos_permanentes[chave] = valor

    def gerar_contexto_permanente(self) -> str:
        if not self.fatos_permanentes:
            return ""
        linhas = ["[Informações permanentes do usuário]"]
        for chave, valor in self.fatos_permanentes.items():
            linhas.append(f"- {chave}: {valor}")
        return "\n".join(linhas)
```

---

## 4.5 Compressão de Contexto

Quando o contexto cresce demais, compressão é necessária. Existem técnicas com diferentes tradeoffs entre fidelidade e redução.

### KV Cache: Por Que Prefixos Estáveis Importam

Os modelos modernos usam KV (Key-Value) cache para evitar reprocessar tokens que não mudaram. Se o seu system prompt e o início do histórico são estáveis entre requisições, o modelo pode reutilizar o cache — reduzindo latência e custo.

```python
# Bom: prefixo estável (system prompt fixo no início)
mensagens_boas = [
    {"role": "system", "content": SYSTEM_PROMPT_FIXO},     # sempre igual
    {"role": "user", "content": "turn 1"},                 # sempre igual
    {"role": "assistant", "content": "resposta 1"},        # sempre igual
    {"role": "user", "content": "nova mensagem"},          # muda a cada turn
]

# Ruim: prefixo instável (timestamp no system prompt)
mensagens_ruins = [
    {"role": "system", "content": f"Hora atual: {datetime.now()}. {SYSTEM_PROMPT}"},
    # ^ invalida o cache a cada requisição
]
```

### Pipeline de Compressão de Contexto

```python
from openai import OpenAI
import tiktoken
from dataclasses import dataclass

client = OpenAI()

@dataclass
class ContextCompressor:
    """
    Pipeline de compressão de contexto com múltiplas técnicas.
    Aplica técnicas em ordem crescente de agressividade.
    """
    modelo: str = "gpt-4o"
    target_tokens: int = 2000

    def _contar_tokens(self, mensagens: list[dict]) -> int:
        enc = tiktoken.encoding_for_model(self.modelo)
        return sum(len(enc.encode(m["content"])) for m in mensagens)

    def comprimir(self, mensagens: list[dict]) -> list[dict]:
        """Aplica compressão até atingir o target de tokens."""
        tokens_atuais = self._contar_tokens(mensagens)

        if tokens_atuais <= self.target_tokens:
            return mensagens  # já está dentro do limite

        # Técnica 1: Truncação simples (remove os mais antigos)
        comprimidas = self._truncar(mensagens)
        if self._contar_tokens(comprimidas) <= self.target_tokens:
            return comprimidas

        # Técnica 2: Resumo dos turnos mais antigos
        comprimidas = self._resumir_antigos(comprimidas)
        if self._contar_tokens(comprimidas) <= self.target_tokens:
            return comprimidas

        # Técnica 3: Deduplicação semântica
        comprimidas = self._deduplicar(comprimidas)

        return comprimidas

    def _truncar(self, mensagens: list[dict], manter_recentes: int = 6) -> list[dict]:
        """Mantém apenas as N mensagens mais recentes."""
        return mensagens[-manter_recentes:] if len(mensagens) > manter_recentes else mensagens

    def _resumir_antigos(self, mensagens: list[dict], preservar_recentes: int = 4) -> list[dict]:
        """Substitui mensagens antigas por resumo compacto."""
        if len(mensagens) <= preservar_recentes:
            return mensagens

        antigas = mensagens[:-preservar_recentes]
        recentes = mensagens[-preservar_recentes:]

        historico_txt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in antigas)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Resuma esta conversa em no máximo 150 palavras, "
                        f"preservando decisões e fatos importantes:\n\n{historico_txt}"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=250
        )
        resumo = resp.choices[0].message.content
        resumo_msg = {"role": "system", "content": f"[Resumo da conversa anterior]\n{resumo}"}
        return [resumo_msg] + recentes

    def _deduplicar(self, mensagens: list[dict]) -> list[dict]:
        """Remove mensagens com conteúdo muito similar (deduplicação básica)."""
        if len(mensagens) <= 2:
            return mensagens

        resultado = [mensagens[0]]
        for msg in mensagens[1:]:
            # Verifica similaridade com última mensagem do mesmo role
            ultimas_mesmo_role = [
                m for m in resultado if m["role"] == msg["role"]
            ]
            if ultimas_mesmo_role:
                ultima = ultimas_mesmo_role[-1]
                # Similaridade simples por sobreposição de palavras
                palavras_atual = set(msg["content"].lower().split())
                palavras_ultima = set(ultima["content"].lower().split())
                if palavras_atual and palavras_ultima:
                    overlap = len(palavras_atual & palavras_ultima) / len(palavras_atual | palavras_ultima)
                    if overlap > 0.85:  # 85% de sobreposição = duplicata
                        continue
            resultado.append(msg)
        return resultado
```

---

## 4.6 Gerenciamento de Sessão em Produção

Em produção, sessões conversacionais precisam ser gerenciadas com as mesmas práticas de qualquer sistema stateful: identificação, persistência, expiração e limpeza.

### Estrutura de Sessão

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json
import uuid

@dataclass
class Sessao:
    session_id: str
    user_id: str
    criada_em: str
    atualizada_em: str
    historico: list[dict] = field(default_factory=list)
    metadados: dict = field(default_factory=dict)  # preferências, estado, etc.
    resumo: Optional[str] = None

    @classmethod
    def nova(cls, user_id: str) -> "Sessao":
        agora = datetime.now().isoformat()
        return cls(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            criada_em=agora,
            atualizada_em=agora
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Sessao":
        return cls(**json.loads(json_str))

class GerenciadorSessoes:
    """
    Gerenciador de sessões conversacionais com storage plugável.
    Interface simples que pode ser implementada com Redis, DynamoDB, etc.
    """

    def __init__(self, storage, ttl_segundos: int = 3600):
        self.storage = storage
        self.ttl = ttl_segundos

    def _chave(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def criar_sessao(self, user_id: str) -> Sessao:
        sessao = Sessao.nova(user_id)
        self.storage.setex(self._chave(sessao.session_id), self.ttl, sessao.to_json())
        return sessao

    def obter_sessao(self, session_id: str) -> Optional[Sessao]:
        dados = self.storage.get(self._chave(session_id))
        if not dados:
            return None
        return Sessao.from_json(dados)

    def salvar_sessao(self, sessao: Sessao):
        sessao.atualizada_em = datetime.now().isoformat()
        self.storage.setex(self._chave(sessao.session_id), self.ttl, sessao.to_json())

    def encerrar_sessao(self, session_id: str):
        self.storage.delete(self._chave(session_id))

    def obter_ou_criar(self, session_id: Optional[str], user_id: str) -> Sessao:
        """Obtém sessão existente ou cria nova se não encontrada."""
        if session_id:
            sessao = self.obter_sessao(session_id)
            if sessao:
                return sessao
        return self.criar_sessao(user_id)
```

### Consistência de Persona Across Sessions

Um problema real em produção: o usuário fechou o chat, voltou três dias depois, e o sistema tratou como se fosse a primeira conversa. Isso quebra a experiência.

```python
@dataclass
class PerfilUsuario:
    """Informações persistentes do usuário que transcendem sessões individuais."""
    user_id: str
    nome: Optional[str] = None
    preferencias: dict = field(default_factory=dict)
    historico_resumido: list[str] = field(default_factory=list)  # resumos das últimas sessões

def injetar_perfil_no_contexto(perfil: PerfilUsuario, system_prompt: str) -> str:
    """Injeta informações persistentes do usuário no system prompt."""
    if not any([perfil.nome, perfil.preferencias, perfil.historico_resumido]):
        return system_prompt

    contexto_usuario = ["\n\n[Contexto do usuário (persistente entre sessões)]"]
    if perfil.nome:
        contexto_usuario.append(f"Nome: {perfil.nome}")
    if perfil.preferencias:
        for chave, valor in perfil.preferencias.items():
            contexto_usuario.append(f"Preferência - {chave}: {valor}")
    if perfil.historico_resumido:
        contexto_usuario.append("Interações anteriores:")
        for resumo in perfil.historico_resumido[-3:]:  # últimas 3 sessões
            contexto_usuario.append(f"  - {resumo}")

    return system_prompt + "\n".join(contexto_usuario)
```

---

## 4.7 Políticas de Inclusão/Exclusão de Contexto

Nem tudo que aparece numa conversa deve entrar no contexto das requisições seguintes. Políticas de inclusão/exclusão são uma camada de controle crítica em sistemas de produção.

### O Que Nunca Deve Entrar no Contexto

**PII (Personally Identifiable Information):**
- Números de CPF, RG, passaporte
- Números de cartão de crédito
- Senhas ou tokens de autenticação
- Dados de saúde ou financeiros sensíveis

**Segredos do sistema:**
- API keys mencionadas pelo usuário
- Credenciais de banco de dados
- Informações sobre infraestrutura interna

### Motor de Políticas de Contexto

```python
import re
from dataclasses import dataclass
from typing import Callable

@dataclass
class PoliticaContexto:
    nome: str
    descricao: str
    detector: Callable[[str], bool]
    acao: str  # "bloquear", "sanitizar", "logar"
    substituto: str = "[REDACTED]"

class MotorPoliticas:
    """
    Aplica políticas de inclusão/exclusão ao conteúdo do contexto.
    Executa antes de adicionar mensagens ao histórico.
    """

    POLITICAS_PADRAO: list[PoliticaContexto] = [
        PoliticaContexto(
            nome="cpf",
            descricao="Remove CPFs do contexto",
            detector=lambda t: bool(re.search(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}', t)),
            acao="sanitizar",
            substituto="[CPF_REDACTED]"
        ),
        PoliticaContexto(
            nome="cartao_credito",
            descricao="Remove números de cartão de crédito",
            detector=lambda t: bool(re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', t)),
            acao="sanitizar",
            substituto="[CARD_REDACTED]"
        ),
        PoliticaContexto(
            nome="api_key",
            descricao="Remove API keys (padrão OpenAI)",
            detector=lambda t: bool(re.search(r'sk-[a-zA-Z0-9]{20,}', t)),
            acao="bloquear",
            substituto="[API_KEY_REDACTED]"
        ),
        PoliticaContexto(
            nome="senha_explicita",
            descricao="Detecta senhas enviadas explicitamente",
            detector=lambda t: bool(re.search(
                r'(?:senha|password|pwd)\s*[:=]\s*\S+', t, re.IGNORECASE
            )),
            acao="sanitizar",
            substituto="[PASSWORD_REDACTED]"
        ),
    ]

    def __init__(self, politicas: list[PoliticaContexto] | None = None):
        self.politicas = politicas or self.POLITICAS_PADRAO

    def processar(self, conteudo: str) -> tuple[str, list[str]]:
        """
        Aplica todas as políticas ao conteúdo.
        Retorna (conteudo_processado, lista_de_politicas_disparadas).
        """
        resultado = conteudo
        politicas_disparadas = []

        for politica in self.politicas:
            if politica.detector(resultado):
                politicas_disparadas.append(politica.nome)

                if politica.acao == "sanitizar":
                    resultado = self._sanitizar(resultado, politica)
                elif politica.acao == "bloquear":
                    # Em vez de bloquear totalmente, sanitiza e loga
                    resultado = self._sanitizar(resultado, politica)
                    print(f"[POLITICA] {politica.nome} disparada — conteúdo sanitizado")

        return resultado, politicas_disparadas

    def _sanitizar(self, texto: str, politica: PoliticaContexto) -> str:
        """Substitui padrão detectado pelo substituto definido na política."""
        # Re-usa o regex do detector para substituição
        # Para produção: refinar com regex específico por política
        return re.sub(
            self._extrair_pattern(politica),
            politica.substituto,
            texto,
            flags=re.IGNORECASE
        )

    def _extrair_pattern(self, politica: PoliticaContexto) -> str:
        """Extrai o padrão regex da função detector (heurística)."""
        # Em produção: cada política deve ter um regex explícito separado
        PATTERNS = {
            "cpf": r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}',
            "cartao_credito": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "api_key": r'sk-[a-zA-Z0-9]{20,}',
            "senha_explicita": r'(?:senha|password|pwd)\s*[:=]\s*\S+',
        }
        return PATTERNS.get(politica.nome, r'.')

# Exemplo de uso
motor = MotorPoliticas()

mensagem_com_dados = (
    "Meu CPF é 123.456.789-09 e meu cartão é 4532 1234 5678 9012. "
    "A API key que uso é sk-proj-abc123def456ghi789jkl012. "
    "Preciso de ajuda para fazer uma compra."
)

conteudo_limpo, disparadas = motor.processar(mensagem_com_dados)
print(f"Políticas disparadas: {disparadas}")
print(f"Conteúdo limpo: {conteudo_limpo}")
```

### Filtragem por Relevância por Turno

```python
def filtrar_contexto_por_relevancia(
    historico: list[dict],
    topico_atual: str,
    max_mensagens: int = 10
) -> list[dict]:
    """
    Filtra o histórico mantendo apenas o que é relevante para o tópico atual.
    Combina recência com relevância semântica básica.
    """
    # Sempre inclui as últimas mensagens (recência)
    recentes = historico[-4:] if len(historico) > 4 else historico

    # Para o restante, filtra por tópico (simplificado: por palavras-chave)
    topico_palavras = set(topico_atual.lower().split())
    antigas = historico[:-4] if len(historico) > 4 else []

    relevantes = []
    for msg in antigas:
        palavras_msg = set(msg["content"].lower().split())
        overlap = len(topico_palavras & palavras_msg)
        if overlap >= 2:  # pelo menos 2 palavras em comum
            relevantes.append(msg)

    # Combina relevantes + recentes, respeita limite
    combinadas = relevantes + recentes
    return combinadas[-max_mensagens:]
```

---

## 4.8 Anti-Padrões Comuns

Reconhecer anti-padrões é tão importante quanto conhecer boas práticas. Em projetos reais, esses erros aparecem frequentemente.

### Anti-padrão 1: O "Kitchen Sink" Context

Incluir tudo no contexto esperando que o modelo filtre o que é relevante.

```python
# ANTI-PADRÃO: envia todo o histórico sem critério
def chat_kitchen_sink(user_msg: str, todo_historico: list[dict]) -> str:
    mensagens = [{"role": "system", "content": SYSTEM}]
    mensagens.extend(todo_historico)  # <- pode ser 500 turnos = 200k tokens
    mensagens.append({"role": "user", "content": user_msg})
    # Resultado: custo explodindo, latência alta, e o modelo pode se perder
    return client.chat.completions.create(model="gpt-4o", messages=mensagens).choices[0].message.content

# PADRÃO CORRETO: selecionar o que realmente é necessário
def chat_seletivo(user_msg: str, historico: list[dict]) -> str:
    contexto = selecionar_por_recencia(historico, max_turnos=5)
    mensagens = [{"role": "system", "content": SYSTEM}]
    mensagens.extend(contexto)
    mensagens.append({"role": "user", "content": user_msg})
    return client.chat.completions.create(model="gpt-4o", messages=mensagens).choices[0].message.content
```

### Anti-padrão 2: Contexto Stale (Informação Obsoleta)

Manter no contexto informações que não são mais válidas.

```
Turno 3: "Meu orçamento é R$ 5.000"
Turno 7: "Recebi uma promoção, agora posso gastar até R$ 15.000"
Turno 15: O contexto ainda inclui o turno 3 com o orçamento antigo
```

**Solução:** Ao detectar atualização de fato, remova a versão antiga do contexto ou marque como obsoleta.

```python
def atualizar_fato_no_contexto(historico: list[dict], chave: str, novo_valor: str) -> list[dict]:
    """Remove menção anterior de um fato e registra o novo valor."""
    # Em produção: use entidades explícitas em vez de busca por texto
    historico_limpo = []
    for msg in historico:
        # Marca mensagens antigas sobre o mesmo fato como obsoletas
        if chave.lower() in msg["content"].lower():
            msg = {**msg, "content": f"[OBSOLETO] {msg['content']}"}
        historico_limpo.append(msg)
    historico_limpo.append({"role": "system", "content": f"ATUALIZAÇÃO: {chave} = {novo_valor}"})
    return historico_limpo
```

### Anti-padrão 3: Context Poisoning

Uma informação incorreta que contamina todas as respostas subsequentes.

```
Turno 1: Usuário fornece dado errado (ex: nome incorreto de uma função)
Turnos 2-20: Modelo usa esse dado errado em todas as respostas
```

**Mitigações:**
- Validar inputs críticos antes de adicionar ao contexto
- Incluir instrução no system prompt: "Se o usuário fornecer informação que contradiz fatos verificáveis, corrija educadamente"
- Implementar checkpoint de verificação para dados críticos

### Anti-padrão 4: Memória Infinita

Acumular histórico indefinidamente sem estratégia de compressão.

| Turns | Tokens (estimado) | Custo por turn (gpt-4o) |
|-------|------------------|------------------------|
| 10 | ~2.000 | ~$0.01 |
| 50 | ~12.000 | ~$0.06 |
| 200 | ~50.000 | ~$0.25 |
| 500 | ~130.000 | ~$0.65 |

Em um sistema com 10.000 usuários ativos com conversas longas, o custo explode. A regra: **implemente compressão desde o início**, não quando o custo já for problemático.

```python
# Resumo de limites operacionais recomendados
LIMITES_RECOMENDADOS = {
    "chatbot_simples": {
        "max_turns_verbatim": 10,
        "estrategia": "truncação",
        "tokens_max": 2000
    },
    "assistente_tecnico": {
        "max_turns_verbatim": 6,
        "estrategia": "resumo_rolante",
        "tokens_max": 4000
    },
    "agente_complexo": {
        "max_turns_verbatim": 4,
        "estrategia": "resumo_rolante + retrieval",
        "tokens_max": 6000
    }
}
```

### Tabela de Anti-Padrões

| Anti-padrão | Sintoma | Custo | Solução |
|-------------|---------|-------|---------|
| Kitchen sink context | Custo cresce linearmente com turns | Alto | Seleção por relevância + recência |
| Contexto stale | Respostas baseadas em dados obsoletos | Médio | Atualizar fatos explicitamente |
| Context poisoning | Erro se propaga por toda conversa | Médio | Validação de input, checkpoint |
| Memória infinita | Custo explode em conversas longas | Alto | Compressão proativa |
| Sem política de PII | Dados sensíveis ficam no contexto | Crítico | Motor de políticas obrigatório |

---

## Resumo da Parte 04

| Conceito | Definição |
|----------|-----------|
| Memória stateless | Todo histórico é enviado pelo cliente em cada requisição; simples, escalável |
| Memória stateful | Servidor mantém sessão; menor payload, maior complexidade |
| Rolling summary | Resumo automático de turnos antigos preserva contexto com custo controlado |
| KV cache | Prefixo estável entre requisições reduz custo e latência |
| Motor de políticas | Filtragem de PII e dados sensíveis antes de entrar no contexto |
| Compressão de contexto | Truncação → resumo → deduplicação, em ordem crescente de agressividade |
| Anti-padrão kitchen sink | Incluir todo o histórico sem filtro — custo e latência altos sem ganho proporcional |
| Memória infinita | Acúmulo sem compressão — a regra é implementar estratégia desde o início |

## 🔗 Referências

- [OpenAI — Managing conversation history](https://platform.openai.com/docs/guides/conversation-state)
- [Anthropic — Long context tips](https://docs.anthropic.com/en/docs/build-with-claude/long-context-tips)
- [tiktoken — OpenAI tokenizer](https://github.com/openai/tiktoken)
- [Redis — Session storage patterns](https://redis.io/docs/latest/develop/use/patterns/session-management/)
- [LangChain — Memory](https://python.langchain.com/docs/concepts/memory/)
- [OWASP LLM Top 10 — Sensitive Information Disclosure](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---
⬅️ **Anterior:** [Parte 03](./parte-03-engenharia-de-contexto-1.md) | ➡️ **Próximo:** [Parte 05](./parte-05-conhecimento-externo-rag.md)  
🏠 **Início:** [README](../README.md)
