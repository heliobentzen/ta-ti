# Parte 10 — Projeto Final e Tendências

> **Carga horária:** 3 horas  
> **Prática correspondente:** [Prática 10](../praticas/pratica-10-projeto-final.md)

---

## 10.1 Consolidando o Aprendizado

Chegamos ao final da disciplina. Você aprendeu:

| Parte | Conceito | Status |
|-------|----------|--------|
| 01 | Introdução à IA Generativa | ✅ |
| 02 | Como funcionam os LLMs | ✅ |
| 03 | APIs de LLMs | ✅ |
| 04 | Prompt Engineering | ✅ |
| 05 | Embeddings | ✅ |
| 06 | Bancos Vetoriais | ✅ |
| 07 | RAG | ✅ |
| 08 | Agentes | ✅ |
| 09 | Ferramentas com IA | ✅ |
| 10 | **Projeto Final + Tendências** | 🎯 |

---

## 10.2 O Projeto Final

O projeto final integra **todos os conceitos** em um **Assistente Inteligente** completo:

### Especificação

Construir um assistente de IA para um domínio à sua escolha com:

1. **Base de conhecimento** (RAG com documentos do domínio)
2. **Conversação com memória** (histórico de chat)
3. **Ferramentas** (pelo menos 2 ferramentas relevantes)
4. **Interface** (CLI, API REST ou interface web)
5. **Avaliação** (métricas de qualidade)

### Domínios Sugeridos

- 📚 Assistente de estudos (responde perguntas sobre o curso)
- 🏥 Assistente médico (apenas informativo, sem diagnóstico)
- ⚖️ Assistente jurídico (consulta de leis e regulamentos)
- 🏢 Assistente de RH (políticas e procedimentos da empresa)
- 💻 Assistente de código (documentação e boas práticas)
- 🎓 Tutor de programação (exercícios e explicações)

---

## 10.3 Arquitetura de Referência

```
┌─────────────────────────────────────────────────────────┐
│                    ASSISTENTE INTELIGENTE                │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Interface │ → │    Agente    │ ← │  Ferramentas │  │
│  │ (web/CLI) │    │   Central    │    │  - Busca web │  │
│  └──────────┘    │              │    │  - Calculadora│  │
│                  │  ┌────────┐  │    │  - API externa│  │
│  ┌──────────┐    │  │  LLM   │  │    └──────────────┘  │
│  │  Memória │ ↔  │  │GPT-4o  │  │                      │
│  │(histórico│    │  └────────┘  │    ┌──────────────┐  │
│  │  + estado│    │       ↑      │ ← │  Base RAG    │  │
│  └──────────┘    └──────────────┘    │ (ChromaDB)   │  │
│                                      └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 10.4 Implementação de Referência

```python
# assistente_final.py
import json
from openai import OpenAI
import chromadb
from datetime import datetime

class AssistenteInteligente:
    """
    Assistente completo com RAG, ferramentas e memória de conversação.
    """
    
    def __init__(self, nome: str, dominio: str, system_prompt: str):
        self.nome = nome
        self.dominio = dominio
        self.client = OpenAI()
        
        # Banco vetorial para RAG
        chroma = chromadb.PersistentClient(path=f"./data/{nome}")
        self.kb = chroma.get_or_create_collection(
            "knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Histórico de conversação
        self.history = []
        self.system_prompt = system_prompt
        
        # Registro de uso
        self.usage_log = []
    
    # ─────────────────────────────
    # GESTÃO DO CONHECIMENTO (RAG)
    # ─────────────────────────────
    
    def adicionar_conhecimento(self, textos: list[str], fontes: list[str]):
        """Adiciona documentos à base de conhecimento."""
        ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(textos))]
        metadatas = [{"fonte": s} for s in fontes]
        self.kb.add(documents=textos, ids=ids, metadatas=metadatas)
        print(f"✅ {len(textos)} documentos adicionados à base de conhecimento")
    
    def buscar_conhecimento(self, query: str, n: int = 3) -> str:
        """Busca informações relevantes na base de conhecimento."""
        if self.kb.count() == 0:
            return ""
        
        results = self.kb.query(query_texts=[query], n_results=min(n, self.kb.count()))
        docs = results["documents"][0]
        sources = [m.get("fonte", "?") for m in results["metadatas"][0]]
        
        return "\n\n".join([f"[{src}]\n{doc}" for doc, src in zip(docs, sources)])
    
    # ──────────────────────────────
    # FERRAMENTAS
    # ──────────────────────────────
    
    def _get_tools_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "buscar_na_base",
                    "description": "Busca informações específicas na base de conhecimento",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "consulta": {"type": "string", "description": "O que buscar"}
                        },
                        "required": ["consulta"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "registrar_informacao",
                    "description": "Salva uma informação importante mencionada pelo usuário",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chave": {"type": "string"},
                            "valor": {"type": "string"}
                        },
                        "required": ["chave", "valor"]
                    }
                }
            }
        ]
    
    def _executar_ferramenta(self, nome: str, args: dict) -> str:
        if nome == "buscar_na_base":
            resultado = self.buscar_conhecimento(args["consulta"])
            return resultado or "Nenhuma informação encontrada sobre isso."
        
        elif nome == "registrar_informacao":
            self._estado = getattr(self, "_estado", {})
            self._estado[args["chave"]] = args["valor"]
            return f"Informação '{args['chave']}' registrada."
        
        return "Ferramenta desconhecida"
    
    # ──────────────────────────────
    # CONVERSA PRINCIPAL
    # ──────────────────────────────
    
    def chat(self, mensagem: str) -> str:
        """Processa uma mensagem e retorna a resposta do assistente."""
        
        # Buscar contexto RAG automaticamente
        contexto_rag = self.buscar_conhecimento(mensagem)
        
        # Construir system prompt com contexto
        system = self.system_prompt
        if contexto_rag:
            system += f"\n\nINFORMAÇÕES RELEVANTES DA BASE DE CONHECIMENTO:\n{contexto_rag}"
        
        # Montar mensagens
        messages = [{"role": "system", "content": system}] + \
                   self.history[-10:] + \
                   [{"role": "user", "content": mensagem}]
        
        # Loop do agente
        for _ in range(5):  # máximo 5 passos
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self._get_tools_schema(),
                tool_choice="auto",
                temperature=0.3
            )
            
            msg = response.choices[0].message
            
            # Registrar uso
            self.usage_log.append({
                "ts": datetime.now().isoformat(),
                "tokens": response.usage.total_tokens
            })
            
            if response.choices[0].finish_reason == "stop":
                # Atualizar histórico
                self.history.append({"role": "user", "content": mensagem})
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content
            
            if response.choices[0].finish_reason == "tool_calls":
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    resultado = self._executar_ferramenta(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": resultado
                    })
        
        return "Não consegui processar sua solicitação."
    
    def limpar_historico(self):
        self.history = []
        print("🗑️  Histórico limpo")
    
    def estatisticas(self) -> dict:
        total_tokens = sum(u["tokens"] for u in self.usage_log)
        return {
            "total_interacoes": len(self.usage_log),
            "total_tokens": total_tokens,
            "custo_estimado_usd": total_tokens * 0.00015 / 1000,
            "documentos_na_base": self.kb.count()
        }


# ──────────────────────────────────
# EXEMPLO DE USO
# ──────────────────────────────────

if __name__ == "__main__":
    assistente = AssistenteInteligente(
        nome="assistente_curso",
        dominio="educação",
        system_prompt="""Você é um assistente educacional especializado em IA Generativa
para o curso de Tópicos Avançados em TI do IFPE.
Seja didático, use exemplos práticos e encoraje os alunos.
Quando não souber algo, admita e sugira onde buscar."""
    )
    
    # Adicionar conhecimento
    assistente.adicionar_conhecimento(
        textos=[
            "RAG significa Retrieval Augmented Generation. É uma técnica que combina busca em documentos com geração de texto por LLMs.",
            "Embeddings são representações vetoriais de texto. Textos similares têm vetores próximos no espaço vetorial.",
            "LLMs são modelos de linguagem de grande escala, treinados em bilhões de textos para prever o próximo token.",
        ],
        fontes=["parte-07.md", "parte-05.md", "parte-02.md"]
    )
    
    # Interface de linha de comando
    print(f"\n🤖 {assistente.nome} iniciado! Digite 'sair' para encerrar.\n")
    
    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in ["sair", "exit", "quit"]:
            stats = assistente.estatisticas()
            print(f"\n📊 Estatísticas: {stats}")
            break
        if not user_input:
            continue
        
        resposta = assistente.chat(user_input)
        print(f"\n🤖 Assistente: {resposta}\n")
```

---

## 10.5 Critérios de Avaliação do Projeto

| Critério | Peso | Descrição |
|---------|------|-----------|
| Funcionalidade | 30% | O assistente funciona conforme especificado? |
| RAG implementado | 20% | Base de conhecimento indexada e buscada corretamente? |
| Ferramentas | 15% | Pelo menos 2 ferramentas funcionando? |
| Qualidade do código | 15% | Código limpo, organizado e com tratamento de erros? |
| Interface | 10% | Interface usável (CLI, API ou web)? |
| Apresentação | 10% | Demonstração clara do funcionamento? |

---

## 10.6 Tendências e Próximos Passos

### O que está acontecendo agora (2024-2025)

#### 1. Modelos Menores e Mais Eficientes

A tendência não é só "maior = melhor":
- **Phi-3 Mini** (3.8B parâmetros) supera modelos 10x maiores em benchmarks específicos
- **Llama 3.2 1B/3B**: modelos que rodam no celular
- **Quantização**: rodar modelos grandes em hardware comum (4-bit, 8-bit)

#### 2. Raciocínio Avançado (Chain of Thought Nativo)

- **OpenAI o1/o3**: modelos que "pensam" antes de responder
- **DeepSeek R1**: open-source com raciocínio comparável ao o1
- Melhorias significativas em matemática, código e lógica

#### 3. Multimodalidade

- **Visão**: modelos que analisam imagens e documentos
- **Áudio**: transcrição e geração de fala integradas
- **Vídeo**: análise e geração de vídeos curtos

#### 4. Agentes Cada Vez Mais Autônomos

- **Computer Use** (Anthropic): agente que controla o computador
- **Operator** (OpenAI): agente que navega na web e executa tarefas
- **Google Workspace AI**: automação integrada no Gmail, Docs, etc.

#### 5. IA no Edge (Dispositivo Local)

- Modelos rodando diretamente no iPhone, Android
- Sem latência de rede, sem custo de API, privacidade total
- **Apple Intelligence**, **Google Gemini Nano**

### Skills para Desenvolver

```
NÍVEL ATUAL (você agora):
✅ Usar APIs de LLMs
✅ Prompt Engineering
✅ RAG básico e avançado
✅ Agentes simples
✅ Integrar IA em aplicações

PRÓXIMO NÍVEL:
→ Fine-tuning de modelos (LoRA, QLoRA)
→ Avaliação sistemática (LLM-as-judge)
→ MLOps para LLMs (monitoramento, re-treino)
→ Multi-agente complexo (LangGraph avançado)
→ Modelos multimodais (visão + texto)
→ IA em produção (latência, custo, confiabilidade)
```

---

## 10.7 Recursos para Continuar Aprendendo

### Cursos e Plataformas

| Recurso | Foco | Gratuito? |
|---------|------|-----------|
| [fast.ai](https://fast.ai) | ML prático | ✅ |
| [DeepLearning.AI](https://deeplearning.ai) | Andrew Ng, LLMs | Parcial |
| [Hugging Face Course](https://huggingface.co/learn) | Transformers, NLP | ✅ |
| [LangChain Academy](https://academy.langchain.com) | LangChain/LangGraph | Parcial |
| [Andrej Karpathy (YouTube)](https://youtube.com/@AndrejKarpathy) | Fundamentos de IA | ✅ |

### Newsletters e Blogs

- [The Batch (deeplearning.ai)](https://www.deeplearning.ai/the-batch/)
- [Simon Willison's Weblog](https://simonwillison.net)
- [Ahead of AI](https://magazine.sebastianraschka.com)

### Comunidades

- Hugging Face Discord
- LangChain Discord
- r/MachineLearning
- Papers With Code

---

## 📌 Resumo Final da Disciplina

```
┌─────────────────────────────────────────────┐
│        JORNADA COMPLETA DA DISCIPLINA       │
│                                             │
│  FUNDAMENTOS                                │
│  ├─ O que é IA Generativa                  │
│  ├─ Como LLMs funcionam                    │
│  ├─ APIs de LLMs                           │
│  └─ Prompt Engineering                     │
│                                             │
│  DADOS E MEMÓRIA                           │
│  ├─ Embeddings                             │
│  ├─ Bancos Vetoriais                       │
│  └─ RAG                                    │
│                                             │
│  AGENTES E FERRAMENTAS                     │
│  ├─ Agentes de IA                          │
│  └─ Construindo com IA                     │
│                                             │
│  PROJETO FINAL                             │
│  └─ Assistente Inteligente completo        │
└─────────────────────────────────────────────┘
```

> **Parabéns por concluir a disciplina!** 🎉  
> Você tem agora as ferramentas para construir aplicações poderosas com IA.  
> O melhor aprendizado vem construindo. Continue criando!

---

⬅️ **Anterior:** [Parte 09](./parte-09-ferramentas-com-ia.md)  
🏠 **Início:** [README](../README.md)  
🛠️ **Prática:** [Prática 10 — Projeto Final](../praticas/pratica-10-projeto-final.md)
