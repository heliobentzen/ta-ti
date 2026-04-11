# Prática 04 — Engenharia de Contexto II: Memória, Estado e Políticas de Contexto

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 04](../conteudo/parte-04-engenharia-de-contexto-2.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:

- Implementar diferentes estratégias de memória em chatbots multi-turno
- Medir e controlar o consumo de tokens ao longo de uma conversa
- Aplicar compressão e sumarização de contexto para conversas longas
- Decidir o que entra e o que sai do contexto em cada turno
- Implementar memória persistente com armazenamento externo

---

## 📝 Exercício 1 — Comparando Estratégias de Memória

Implemente um chatbot de atendimento de suporte técnico com **três estratégias distintas de memória** e compare os resultados.

### 1.1 — Memória Completa (Naive)

```python
from openai import OpenAI
import tiktoken

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

def contar_tokens(mensagens: list[dict]) -> int:
    """Conta tokens aproximados na lista de mensagens."""
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in mensagens:
        total += len(enc.encode(msg.get("content", "")))
    return total

class ChatbotMemoriaCompleta:
    """Mantém histórico completo — sem controle de custo."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.historico: list[dict] = [{"role": "system", "content": system_prompt}]

    def chat(self, mensagem: str) -> tuple[str, int]:
        self.historico.append({"role": "user", "content": mensagem})
        resp = client.chat.completions.create(model=MODEL, messages=self.historico)
        resposta = resp.choices[0].message.content
        self.historico.append({"role": "assistant", "content": resposta})
        tokens = contar_tokens(self.historico)
        return resposta, tokens

    def reset(self):
        self.historico = [{"role": "system", "content": self.system_prompt}]
```

### 1.2 — Memória com Janela Deslizante

```python
class ChatbotJanelaDeslizante:
    """Mantém apenas as N últimas trocas."""

    def __init__(self, system_prompt: str, max_trocas: int = 5):
        self.system_prompt = system_prompt
        self.max_trocas = max_trocas
        self.historico: list[dict] = []

    def chat(self, mensagem: str) -> tuple[str, int]:
        self.historico.append({"role": "user", "content": mensagem})
        # Trunca o histórico mantendo apenas as últimas N trocas (2 mensagens = 1 troca)
        if len(self.historico) > self.max_trocas * 2:
            self.historico = self.historico[-(self.max_trocas * 2):]

        contexto = [{"role": "system", "content": self.system_prompt}] + self.historico
        resp = client.chat.completions.create(model=MODEL, messages=contexto)
        resposta = resp.choices[0].message.content
        self.historico.append({"role": "assistant", "content": resposta})
        return resposta, contar_tokens(contexto)
```

### 1.3 — Memória com Sumarização

```python
class ChatbotComSumario:
    """Sumariza conversas antigas ao ultrapassar limite de tokens."""

    LIMITE_TOKENS = 800

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.sumario: str = ""
        self.historico_recente: list[dict] = []

    def _sumarizar(self):
        """Comprime o histórico recente em um sumário."""
        if not self.historico_recente:
            return
        texto = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.historico_recente
        )
        prompt_sumario = f"""Você é um assistente de sumarização.
Resuma a conversa abaixo em no máximo 3 frases, preservando os fatos técnicos importantes:

{texto}

Sumário:"""
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_sumario}],
            temperature=0,
        )
        novo_sumario = resp.choices[0].message.content.strip()
        self.sumario = f"{self.sumario}\n{novo_sumario}".strip() if self.sumario else novo_sumario
        self.historico_recente = []

    def _montar_contexto(self) -> list[dict]:
        mensagens = [{"role": "system", "content": self.system_prompt}]
        if self.sumario:
            mensagens.append(
                {"role": "system", "content": f"Resumo da conversa anterior:\n{self.sumario}"}
            )
        return mensagens + self.historico_recente

    def chat(self, mensagem: str) -> tuple[str, int]:
        self.historico_recente.append({"role": "user", "content": mensagem})
        contexto = self._montar_contexto()

        if contar_tokens(contexto) > self.LIMITE_TOKENS:
            self._sumarizar()
            contexto = self._montar_contexto()

        resp = client.chat.completions.create(model=MODEL, messages=contexto)
        resposta = resp.choices[0].message.content
        self.historico_recente.append({"role": "assistant", "content": resposta})
        return resposta, contar_tokens(contexto)
```

### 1.4 — Teste Comparativo

Simule uma conversa de suporte de 10 turnos e registre o consumo de tokens em cada estratégia:

```python
SYSTEM = """Você é o assistente de suporte da empresa TechCorp.
O usuário está tendo problemas com o software de gestão de estoque v2.3.
Seja objetivo e registre sempre o número do chamado: #45821."""

conversa = [
    "Oi, não consigo fazer login no sistema.",
    "Já tentei redefinir a senha mas continua dando erro.",
    "O erro diz: 'Token de sessão inválido'.",
    "Como faço para limpar os dados de sessão no navegador?",
    "Limpei os cookies mas continua com o mesmo erro.",
    "Estou usando Chrome versão 120.",
    "O sistema funciona no Firefox, só falha no Chrome.",
    "Tem algum plugin do Chrome que pode estar causando isso?",
    "Desativei todos os plugins mas o erro persiste.",
    "Preciso de um resumo do problema para enviar ao meu chefe.",
]

for nome, bot in [
    ("Memória Completa", ChatbotMemoriaCompleta(SYSTEM)),
    ("Janela Deslizante (5 trocas)", ChatbotJanelaDeslizante(SYSTEM, max_trocas=5)),
    ("Com Sumarização", ChatbotComSumario(SYSTEM)),
]:
    print(f"\n{'='*50}")
    print(f"Estratégia: {nome}")
    print('='*50)
    for i, msg in enumerate(conversa, 1):
        resposta, tokens = bot.chat(msg)
        print(f"[Turno {i:02d} | {tokens:4d} tokens] U: {msg[:40]}...")
        print(f"              A: {resposta[:60]}...")
```

**Registre:**
- Tokens usados no turno 1, 5 e 10 para cada estratégia
- O décimo turno pede um "resumo do problema" — qual estratégia responde melhor e por quê?

---

## 📝 Exercício 2 — Políticas de Contexto: O que Entra e o que Sai

Em sistemas de produção, nem tudo que o usuário envia deve ir para o contexto do LLM. Implemente um gerenciador com políticas explícitas.

### 2.1 — Política de Filtragem

```python
import re
from dataclasses import dataclass

@dataclass
class PoliticaContexto:
    """Define regras sobre o que entra no contexto do LLM."""
    max_tokens_por_mensagem: int = 500
    remover_dados_sensiveis: bool = True
    incluir_metadata: bool = False
    max_trocas_historico: int = 8

# Padrões de dados sensíveis a remover
PADROES_SENSIVEIS = [
    (r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', '[CPF]'),            # CPF
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARTÃO]'),  # Cartão de crédito
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\+?55?\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}', '[TELEFONE]'),  # Telefone BR
]

def aplicar_politica(mensagem: str, politica: PoliticaContexto) -> str:
    """Aplica políticas de filtragem a uma mensagem."""
    enc = tiktoken.get_encoding("cl100k_base")

    # 1. Remover dados sensíveis
    if politica.remover_dados_sensiveis:
        for padrao, substituto in PADROES_SENSIVEIS:
            mensagem = re.sub(padrao, substituto, mensagem)

    # 2. Truncar mensagens muito longas
    tokens = enc.encode(mensagem)
    if len(tokens) > politica.max_tokens_por_mensagem:
        mensagem = enc.decode(tokens[:politica.max_tokens_por_mensagem]) + "... [truncado]"

    return mensagem

# Teste
mensagens_teste = [
    "Meu CPF é 123.456.789-00 e preciso de ajuda com minha conta.",
    "Me ligue no (81) 99999-8888 quando puder.",
    "Meu email é usuario@empresa.com, o cartão 4111 1111 1111 1111 foi bloqueado.",
]

politica = PoliticaContexto(remover_dados_sensiveis=True, max_tokens_por_mensagem=200)
print("Mensagens após política de filtragem:")
for msg in mensagens_teste:
    filtrada = aplicar_politica(msg, politica)
    print(f"  Original : {msg}")
    print(f"  Filtrada : {filtrada}")
    print()
```

### 2.2 — Contexto com Relevância

Nem todas as mensagens antigas são igualmente relevantes. Implemente priorização:

```python
class ContextoComRelevancia:
    """
    Gerencia o contexto priorizando mensagens por relevância,
    em vez de apenas manter as mais recentes.
    """

    def __init__(self, system_prompt: str, budget_tokens: int = 1500):
        self.system_prompt = system_prompt
        self.budget_tokens = budget_tokens
        self.historico: list[dict] = []
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _tokens(self, texto: str) -> int:
        return len(self.enc.encode(texto))

    def _montar_contexto_otimizado(self, query: str) -> list[dict]:
        """
        Estratégia: sempre inclui a mensagem mais recente do usuário,
        depois preenche o budget com mensagens anteriores priorizando
        as que contêm palavras-chave similares à query atual.
        """
        budget_restante = self.budget_tokens - self._tokens(self.system_prompt)
        contexto = []

        # Ordenar histórico por relevância em relação à query atual
        palavras_query = set(query.lower().split())

        def score_relevancia(msg: dict) -> int:
            palavras_msg = set(msg["content"].lower().split())
            return len(palavras_query & palavras_msg)

        historico_ordenado = sorted(
            self.historico, key=score_relevancia, reverse=True
        )

        for msg in historico_ordenado:
            custo = self._tokens(msg["content"]) + 10  # overhead de role
            if custo <= budget_restante:
                contexto.append(msg)
                budget_restante -= custo

        # Reordenar pela posição original (para manter coerência temporal)
        indices_originais = {id(m): i for i, m in enumerate(self.historico)}
        contexto.sort(key=lambda m: indices_originais.get(id(m), 0))

        return [{"role": "system", "content": self.system_prompt}] + contexto

    def chat(self, mensagem: str) -> str:
        contexto = self._montar_contexto_otimizado(mensagem)
        contexto.append({"role": "user", "content": mensagem})

        resp = client.chat.completions.create(model=MODEL, messages=contexto)
        resposta = resp.choices[0].message.content

        self.historico.append({"role": "user", "content": mensagem})
        self.historico.append({"role": "assistant", "content": resposta})
        return resposta
```

---

## 📝 Exercício 3 — Memória Persistente com Arquivo JSON

Sistemas reais precisam de memória que sobreviva ao reinício do processo. Implemente memória simples baseada em arquivo.

```python
import json
import os
from datetime import datetime

class MemoriaPersistente:
    """Armazena memória de sessão em arquivo JSON."""

    def __init__(self, session_id: str, system_prompt: str, arquivo: str = "/tmp/memoria.json"):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.arquivo = arquivo
        self._dados: dict = self._carregar()

    def _carregar(self) -> dict:
        """Carrega todos os dados do arquivo."""
        if os.path.exists(self.arquivo):
            with open(self.arquivo) as f:
                return json.load(f)
        return {}

    def _salvar(self):
        """Persiste os dados no arquivo."""
        with open(self.arquivo, "w") as f:
            json.dump(self._dados, f, ensure_ascii=False, indent=2)

    def _sessao(self) -> dict:
        """Retorna ou cria dados para a sessão atual."""
        if self.session_id not in self._dados:
            self._dados[self.session_id] = {
                "criada_em": datetime.now().isoformat(),
                "ultima_atividade": datetime.now().isoformat(),
                "historico": [],
                "sumario": "",
            }
        return self._dados[self.session_id]

    def adicionar(self, role: str, content: str):
        sessao = self._sessao()
        sessao["historico"].append({"role": role, "content": content})
        sessao["ultima_atividade"] = datetime.now().isoformat()
        self._salvar()

    def obter_contexto(self) -> list[dict]:
        sessao = self._sessao()
        mensagens = [{"role": "system", "content": self.system_prompt}]
        if sessao["sumario"]:
            mensagens.append(
                {"role": "system", "content": f"Contexto anterior: {sessao['sumario']}"}
            )
        return mensagens + sessao["historico"][-10:]  # últimas 10 mensagens

    def chat(self, mensagem: str) -> str:
        self.adicionar("user", mensagem)
        contexto = self.obter_contexto()
        resp = client.chat.completions.create(model=MODEL, messages=contexto)
        resposta = resp.choices[0].message.content
        self.adicionar("assistant", resposta)
        return resposta


# --- Teste de persistência ---
SESSION = "usuario_joao_2024"
SYSTEM = "Você é um tutor de Python. Lembre-se do progresso do aluno."

bot = MemoriaPersistente(SESSION, SYSTEM)
print("=== Sessão 1 ===")
print(bot.chat("Estou aprendendo Python. O que são listas?"))
print(bot.chat("Pode dar um exemplo com append?"))

# Simula reinício: cria novo objeto com mesmo session_id
bot2 = MemoriaPersistente(SESSION, SYSTEM)
print("\n=== Sessão 2 (após reinício) ===")
print(bot2.chat("Você lembra o que eu estava estudando?"))
# O bot deve lembrar que você estava estudando listas
```

---

## 🏆 Desafios Opcionais

1. **Dashboard de Tokens**: Crie uma visualização ASCII que mostre o "uso do orçamento de contexto" em tempo real durante uma conversa

2. **Compressão Semântica**: Em vez de truncar por tokens, implemente compressão que remove mensagens de menor relevância usando embeddings para medir similaridade com a query atual

3. **Multi-sessão**: Adapte a `MemoriaPersistente` para suportar múltiplos usuários em paralelo, com isolamento de sessões e expiração automática após 24h

4. **Política de Injeção**: Implemente uma função `enriquecer_contexto` que, a cada turno, injeta automaticamente fatos relevantes de uma base de conhecimento (dicionário simples) com base nas palavras-chave da mensagem do usuário

---

## ✅ Checklist de Entrega

- [ ] Ex01: comparativo das 3 estratégias de memória rodando com 10 turnos
- [ ] Ex01: análise do turno 10 — qual estratégia respondeu melhor ao pedido de resumo?
- [ ] Ex02: política de filtragem removendo CPF, email e telefone corretamente
- [ ] Ex02: contexto com relevância implementado e testado
- [ ] Ex03: memória persistente sobrevivendo ao reinício do processo
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 03](./pratica-03-engenharia-de-contexto-1.md) | ➡️ **Próxima:** [Prática 05 — Conhecimento Externo e RAG](./pratica-05-rag.md)
