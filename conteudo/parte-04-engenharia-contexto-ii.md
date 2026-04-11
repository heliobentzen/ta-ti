# Parte 04 — Engenharia de Contexto II

> **Carga horária:** 3 horas  
> **Prática correspondente:** [Prática 02](../praticas/pratica-02-engenharia-contexto.md)

---

## 4.1 Memória e Estado em Aplicações com LLMs

LLMs são **stateless por natureza** — cada chamada à API é independente. Para que uma aplicação com IA funcione de forma contínua e coerente, o desenvolvedor precisa **gerenciar o que entra e o que sai do contexto** em cada requisição.

> **Princípio fundamental:** O modelo só sabe o que está na janela de contexto. Tudo que não está lá, para o modelo, não existe.

### O Problema da Memória

```
Chamada 1: "Meu nome é Ana"
Chamada 2: "Qual é meu nome?"  ← O modelo não sabe!
```

A solução: injetar o histórico de conversa (ou um resumo dele) em cada nova chamada.

---

## 4.2 Padrões de Gerenciamento de Histórico

### Histórico Completo (Sliding Window)

O padrão mais simples — envia todas as mensagens anteriores:

```python
import openai

historico = [
    {"role": "system", "content": "Você é um assistente técnico."}
]

def conversar(mensagem_usuario: str) -> str:
    historico.append({"role": "user", "content": mensagem_usuario})
    
    resposta = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=historico
    )
    
    msg_assistente = resposta.choices[0].message.content
    historico.append({"role": "assistant", "content": msg_assistente})
    return msg_assistente
```

**Problema:** O histórico cresce indefinidamente e eventualmente estoura a janela de contexto.

### Janela Deslizante com Limite

```python
MAX_MENSAGENS = 20

def conversar_com_limite(mensagem_usuario: str) -> str:
    historico.append({"role": "user", "content": mensagem_usuario})
    
    # Mantém system prompt + últimas N mensagens
    mensagens = [historico[0]] + historico[-MAX_MENSAGENS:]
    
    resposta = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=mensagens
    )
    
    msg_assistente = resposta.choices[0].message.content
    historico.append({"role": "assistant", "content": msg_assistente})
    return msg_assistente
```

### Resumo Progressivo

Quando o histórico fica grande, resuma as mensagens mais antigas:

```python
def resumir_historico(mensagens: list[dict]) -> str:
    """Usa o próprio LLM para resumir o histórico."""
    texto = "\n".join(f"{m['role']}: {m['content']}" for m in mensagens)
    
    resposta = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Resuma os pontos principais desta conversa:\n{texto}"
        }]
    )
    return resposta.choices[0].message.content

def gerenciar_contexto(historico: list[dict], limite: int = 15) -> list[dict]:
    """Mantém o contexto dentro do limite, resumindo o que é antigo."""
    if len(historico) <= limite:
        return historico
    
    # Separa system prompt
    system = historico[0]
    mensagens_antigas = historico[1:-limite]
    mensagens_recentes = historico[-limite:]
    
    resumo = resumir_historico(mensagens_antigas)
    
    return [
        system,
        {"role": "system", "content": f"Resumo da conversa anterior: {resumo}"},
        *mensagens_recentes
    ]
```

---

## 4.3 Políticas de Contexto

Uma **política de contexto** define regras sobre o que deve entrar, o que pode sair e o que nunca deve entrar na janela de contexto.

### Tipos de Informação no Contexto

| Tipo | Exemplo | Persistência |
|------|---------|-------------|
| **Identidade** | System prompt, persona, regras | Sempre presente |
| **Sessão** | Histórico da conversa atual | Duração da sessão |
| **Referência** | Documentos recuperados (RAG) | Por mensagem |
| **Efêmero** | Resultado de ferramenta, cálculo | Uma única vez |

### Exemplo de Política

```python
class PoliticaContexto:
    """Define o que entra e sai do contexto."""
    
    def __init__(self):
        self.max_tokens_total = 8000
        self.max_tokens_historico = 3000
        self.max_tokens_documentos = 2000
        self.max_tokens_ferramentas = 1000
        # Reserva para system prompt + resposta
        self.reserva = 2000
    
    def montar_contexto(
        self,
        system_prompt: str,
        historico: list[dict],
        documentos: list[str] = None,
        resultado_ferramenta: str = None
    ) -> list[dict]:
        """Monta o contexto respeitando os limites."""
        mensagens = [{"role": "system", "content": system_prompt}]
        
        # Adiciona histórico (truncado se necessário)
        historico_truncado = self._truncar(historico, self.max_tokens_historico)
        mensagens.extend(historico_truncado)
        
        # Adiciona documentos recuperados
        if documentos:
            contexto_docs = self._truncar_texto(
                "\n\n".join(documentos),
                self.max_tokens_documentos
            )
            mensagens.append({
                "role": "system",
                "content": f"Documentos relevantes:\n{contexto_docs}"
            })
        
        # Adiciona resultado de ferramenta (efêmero)
        if resultado_ferramenta:
            mensagens.append({
                "role": "system",
                "content": f"Resultado da ferramenta:\n{resultado_ferramenta}"
            })
        
        return mensagens
```

---

## 4.4 Estado Além do Contexto

Nem toda informação deve viver dentro da janela de contexto. Algumas precisam de **armazenamento externo**:

### Memória de Longo Prazo

```python
import json
from pathlib import Path

class MemoriaLongoPrazo:
    """Armazena informações entre sessões."""
    
    def __init__(self, caminho: str = "memoria.json"):
        self.caminho = Path(caminho)
        self.dados = self._carregar()
    
    def _carregar(self) -> dict:
        if self.caminho.exists():
            return json.loads(self.caminho.read_text())
        return {"fatos": [], "preferencias": {}}
    
    def salvar(self):
        self.caminho.write_text(json.dumps(self.dados, indent=2, ensure_ascii=False))
    
    def adicionar_fato(self, fato: str):
        """Armazena um fato sobre o usuário."""
        if fato not in self.dados["fatos"]:
            self.dados["fatos"].append(fato)
            self.salvar()
    
    def definir_preferencia(self, chave: str, valor: str):
        """Armazena uma preferência do usuário."""
        self.dados["preferencias"][chave] = valor
        self.salvar()
    
    def gerar_contexto(self) -> str:
        """Gera um bloco de contexto para injetar no prompt."""
        partes = []
        if self.dados["fatos"]:
            partes.append("Fatos conhecidos sobre o usuário:")
            partes.extend(f"- {f}" for f in self.dados["fatos"])
        if self.dados["preferencias"]:
            partes.append("\nPreferências do usuário:")
            partes.extend(f"- {k}: {v}" for k, v in self.dados["preferencias"].items())
        return "\n".join(partes) if partes else ""
```

### Quando Usar Cada Tipo de Memória

| Cenário | Tipo de Memória | Implementação |
|---------|----------------|---------------|
| Conversa em andamento | Histórico (curto prazo) | Lista de mensagens |
| Preferências do usuário | Longo prazo | Banco de dados / JSON |
| Documentos de referência | Recuperada (RAG) | Busca vetorial |
| Resultado de cálculo | Efêmera | Incluída uma vez no contexto |

---

## 4.5 Estratégias de Injeção de Contexto

O **onde** e **como** você injeta informações no contexto importa tanto quanto **o que** você injeta.

### Posicionamento no Prompt

```
┌──────────────────────────────────────┐
│ SYSTEM PROMPT (identidade + regras)  │  ← Sempre no topo
├──────────────────────────────────────┤
│ MEMÓRIA DE LONGO PRAZO (resumo)     │  ← Se houver
├──────────────────────────────────────┤
│ DOCUMENTOS RECUPERADOS (RAG)        │  ← Mais próximo da pergunta
├──────────────────────────────────────┤
│ HISTÓRICO DA CONVERSA               │  ← Mensagens recentes
├──────────────────────────────────────┤
│ MENSAGEM DO USUÁRIO                 │  ← Pergunta atual
└──────────────────────────────────────┘
```

> **Regra prática:** Informação mais **relevante para a resposta atual** deve ficar mais **próxima da mensagem do usuário**. LLMs tendem a dar mais atenção ao início e ao final do contexto.

### Exemplo Completo de Montagem

```python
def montar_prompt_completo(
    pergunta: str,
    system_prompt: str,
    memoria: MemoriaLongoPrazo,
    historico: list[dict],
    documentos_rag: list[str] = None,
    max_historico: int = 10
) -> list[dict]:
    """Monta um prompt completo com todas as camadas de contexto."""
    
    mensagens = []
    
    # 1. Identidade e regras
    system_content = system_prompt
    
    # 2. Memória de longo prazo
    contexto_memoria = memoria.gerar_contexto()
    if contexto_memoria:
        system_content += f"\n\n{contexto_memoria}"
    
    mensagens.append({"role": "system", "content": system_content})
    
    # 3. Documentos recuperados (se houver)
    if documentos_rag:
        docs_texto = "\n---\n".join(documentos_rag)
        mensagens.append({
            "role": "system",
            "content": f"Use os seguintes documentos como referência:\n{docs_texto}"
        })
    
    # 4. Histórico recente
    mensagens.extend(historico[-max_historico:])
    
    # 5. Pergunta atual
    mensagens.append({"role": "user", "content": pergunta})
    
    return mensagens
```

---

## 4.6 Filtragem e Segurança do Contexto

Nem toda informação deve entrar no contexto. Políticas de filtragem protegem contra:

- **Prompt injection:** Dados do usuário que tentam alterar o comportamento do sistema
- **Vazamento de dados:** Informações sensíveis que não devem ser expostas ao modelo
- **Poluição do contexto:** Informação irrelevante que consome tokens sem agregar valor

### Exemplo de Filtro

```python
import re

class FiltroContexto:
    """Filtra e sanitiza informações antes de incluir no contexto."""
    
    PADROES_PERIGOSOS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(everything|all)",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
    ]
    
    def sanitizar(self, texto: str) -> str:
        """Remove padrões potencialmente perigosos."""
        for padrao in self.PADROES_PERIGOSOS:
            texto = re.sub(padrao, "[FILTRADO]", texto, flags=re.IGNORECASE)
        return texto
    
    def validar_tamanho(self, texto: str, max_chars: int = 10000) -> str:
        """Trunca texto que excede o limite."""
        if len(texto) > max_chars:
            return texto[:max_chars] + "\n[...truncado...]"
        return texto
```

---

## 4.7 Debugging de Contexto

Para desenvolver aplicações robustas, é essencial poder **inspecionar o contexto** que está sendo enviado ao modelo:

```python
def debug_contexto(mensagens: list[dict]):
    """Exibe o contexto formatado para depuração."""
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4o")
    
    total_tokens = 0
    print("=" * 60)
    print("DEBUG — Contexto enviado ao modelo")
    print("=" * 60)
    
    for i, msg in enumerate(mensagens):
        tokens = len(encoding.encode(msg["content"]))
        total_tokens += tokens
        role = msg["role"].upper()
        preview = msg["content"][:100].replace("\n", " ")
        print(f"\n[{i}] {role} ({tokens} tokens)")
        print(f"    {preview}...")
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_tokens} tokens em {len(mensagens)} mensagens")
    print(f"{'=' * 60}")
```

---

## 📌 Resumo da Parte 04

| Conceito | Descrição |
|----------|-----------|
| Estado / Memória | LLMs são stateless; estado é responsabilidade do desenvolvedor |
| Histórico de conversa | Injetar mensagens anteriores para continuidade |
| Janela deslizante | Manter apenas N mensagens recentes |
| Resumo progressivo | Usar o LLM para comprimir histórico antigo |
| Política de contexto | Regras sobre o que entra, permanece e sai do contexto |
| Memória de longo prazo | Armazenamento persistente de fatos e preferências |
| Injeção de contexto | Posicionamento estratégico das informações no prompt |
| Filtragem | Sanitização contra injection e poluição do contexto |

---

## 🔗 Referências

- [OpenAI — Managing Conversation History](https://platform.openai.com/docs/guides/conversation)
- [LangChain — Memory](https://python.langchain.com/docs/modules/memory/)
- [Prompt Injection — OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Building LLM-Powered Applications (Chip Huyen)](https://huyenchip.com/2023/04/11/llm-engineering.html)

---

⬅️ **Anterior:** [Parte 03 — Engenharia de Contexto I](./parte-03-engenharia-contexto-i.md) | ➡️ **Próximo:** [Parte 05 — Conhecimento Externo e RAG](./parte-05-conhecimento-externo-rag.md)
