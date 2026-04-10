# Tópicos Avançados em TI — IA Generativa para Programadores

> **Disciplina:** Tópicos Avançados em TI  
> **Instituição:** IFPE — Instituto Federal de Pernambuco  
> **Professor:** Hélio Bentzen  
> **Carga Horária:** 26 horas  

---

## 🎯 Objetivo da Disciplina

Esta disciplina tem como objetivo capacitar programadores a **entender e construir ferramentas práticas com Inteligência Artificial Generativa**, utilizando ferramentas **gratuitas e open-source**. Ao final do curso, o aluno será capaz de:

- Compreender como funcionam os Grandes Modelos de Linguagem (LLMs), incluindo experimentação local
- Integrar APIs de IA (OpenAI, Anthropic, Google) e modelos open-source (Llama, Phi, Mistral) via Ollama
- Criar pipelines de RAG (Retrieval Augmented Generation)
- Desenvolver agentes de IA autônomos com frameworks open-source (smolagents, LangGraph, LangChain)
- Trabalhar com embeddings e bancos de dados vetoriais
- Utilizar ferramentas gratuitas de coding com IA (Aider, Continue.dev, OpenCode)
- Monitorar aplicações de IA com Langfuse (open-source)
- Construir assistentes inteligentes e ferramentas com IA de ponta a ponta

> **📌 Nota:** Esta disciplina é projetada para uso educacional. Todas as ferramentas e frameworks utilizados possuem opções **gratuitas ou open-source**. Modelos locais via [Ollama](https://ollama.ai) são utilizados como alternativa sem custo às APIs comerciais.

---

## 📋 Ementa

| # | Tema | Carga Horária |
|---|------|---------------|
| 01 | Introdução à IA Generativa | 2h |
| 02 | LLMs: Como Funcionam (com experimentação local via Ollama) | 3h |
| 03 | Prompt Engineering (incluindo janela de contexto) | 3h |
| 04 | Embeddings, Bancos Vetoriais e RAG | 4h |
| 05 | **Agentes de IA (smolagents, LangGraph, LangChain)** | **7h** |
| 06 | **Construindo Ferramentas com IA (Langfuse, Aider, Continue.dev)** | **7h** |
| **Total** | | **26h** |

> **📌 Foco prático:** As cargas horárias estão concentradas nos módulos 05 e 06, que cobrem os temas mais práticos e relevantes para o mercado.

---

## 📁 Estrutura do Repositório

```
ta-ti/
├── README.md                  ← Você está aqui
│
├── conteudo/                  ← Material teórico da disciplina
│   ├── parte-01-introducao-ia-generativa.md
│   ├── parte-02-llms-como-funcionam.md
│   ├── parte-03-prompt-engineering.md
│   ├── parte-04-embeddings-vetores-rag.md
│   ├── parte-05-agentes-ia.md
│   └── parte-06-ferramentas-com-ia.md
│
└── praticas/                  ← Atividades práticas evolutivas
    ├── pratica-01-primeiros-passos-llm.md
    ├── pratica-02-prompt-engineering.md
    ├── pratica-03-embeddings-vetores-rag.md
    ├── pratica-04-agente-simples.md
    ├── pratica-05-agente-ferramentas.md
    └── pratica-06-pipeline-projeto-final.md
```

---

## 🚀 Como Usar Este Repositório

### Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.ai) para rodar modelos localmente (gratuito — **principal ferramenta do curso**)
- Familiaridade básica com programação (qualquer linguagem)
- `pip` ou `conda` para gerenciar pacotes
- (Opcional) Conta na [OpenAI Platform](https://platform.openai.com) ou outro provedor de API

### Configuração Inicial

```bash
# Clone o repositório
git clone https://github.com/heliobentzen/ta-ti.git
cd ta-ti

# Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Instale as dependências básicas (serão detalhadas em cada prática)
pip install openai python-dotenv

# Instale o Ollama e baixe um modelo (gratuito)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

### Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Para uso com Ollama (padrão do curso — gratuito):
OLLAMA_HOST=http://localhost:11434

# Para uso com APIs comerciais (opcional):
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

> **💡 Dica:** A maioria dos exercícios do curso pode ser realizada usando apenas o **Ollama** com modelos open-source, sem necessidade de API keys pagas.

---

## 📚 Conteúdo

Acesse a pasta [`conteudo/`](./conteudo/) para o material teórico completo, dividido em 6 partes progressivas.

## 🛠️ Práticas

Acesse a pasta [`praticas/`](./praticas/) para as 6 atividades práticas evolutivas, que vão de uma simples chamada à API até a construção de um assistente inteligente completo.

---

## 🗺️ Trilha de Aprendizado

```
[01 Intro] → [02 LLMs] → [03 Prompts]
                               ↓
                    [04 Embeddings/Vetores/RAG]
                               ↓
                    [05 Agentes de IA ★]
                               ↓
                    [06 Ferramentas com IA ★]
```

> **★ Módulos com maior carga horária** — foco prático e aplicado.

**Cada parte teórica é seguida de uma prática correspondente** para fixação e aplicação imediata do conteúdo.

---

## 💡 Metodologia

- **Aulas expositivas curtas** com foco em conceitos essenciais
- **Demonstrações ao vivo** com código real
- **Práticas hands-on** com problemas progressivos
- **Projeto final integrado** que une todos os conceitos

---

## 📬 Contato

- **Professor:** Hélio Bentzen
- **Instituição:** IFPE — Campus Recife
- **Disciplina:** Tópicos Avançados em TI

---

> *"A melhor forma de entender IA é construindo com ela."*
