# Tópicos Avançados em TI — IA Generativa para Programadores

> **Disciplina:** Tópicos Avançados em TI  
> **Instituição:** IFPE — Instituto Federal de Pernambuco  
> **Professor:** Hélio Bentzen  
> **Carga Horária:** 26 horas  

---

## 🎯 Objetivo da Disciplina

Esta disciplina tem como objetivo capacitar programadores a **entender e construir soluções práticas com IA Generativa**, com foco no que é relevante para o mercado agora e nos próximos anos. Ao final do curso, o aluno será capaz de:

- Entender a arquitetura básica de LLMs e executar modelos localmente via Ollama
- Aplicar técnicas avançadas de prompting e formatar saídas estruturadas (JSON/Markdown)
- Consumir APIs de IA via código e gerenciar estado/histórico de conversação
- Implementar RAG como ferramenta de contexto dinâmico para agentes
- Construir agentes autônomos com Function Calling, LangGraph e smolagents
- Monitorar aplicações de IA com rastreabilidade, custos e telemetria (Langfuse)
- Mitigar riscos como Prompt Injection em ambientes de produção
- Integrar assistentes de codificação com IA (Claude Code e alternativas open-source) ao fluxo de desenvolvimento diário

> **📌 Nota:** Esta disciplina é projetada para uso educacional. Todas as ferramentas e frameworks utilizados possuem opções **gratuitas ou open-source**. Modelos locais via [Ollama](https://ollama.ai) são utilizados como alternativa sem custo às APIs comerciais.

---

## 📋 Ementa

**IA Generativa para Desenvolvimento de Software — 26h**

| # | Módulo | Carga Horária |
|---|--------|---------------|
| 01 | **Fundamentos e Prompt Engineering** | 5h |
| 02 | **Integração e Gerenciamento de Estado** | 4h |
| 03 | **O Essencial de RAG e Contexto** | 3h |
| 04 | **Orquestração e Agentes Autônomos** | **10h** |
| 05 | **Produção, Observabilidade e Ferramental** | 4h |
| **Total** | | **26h** |

### Detalhamento dos Módulos

**Módulo 01 — Fundamentos e Prompt Engineering (5h)**
Arquitetura básica de LLMs. Execução de modelos locais (Ollama). Técnicas avançadas de prompting. Output parsing (formatação de saída em JSON/Markdown) e limites de contexto.

**Módulo 02 — Integração e Gerenciamento de Estado (4h)**
Consumo de APIs de IA via código. Resolução da natureza stateless das LLMs e injeção de memória/histórico de conversação.

**Módulo 03 — O Essencial de RAG e Contexto (3h)**
Conceitos básicos de Embeddings e busca semântica. O RAG estruturado exclusivamente como uma ferramenta de consulta (Tool) para municiar a IA com contexto dinâmico.

**Módulo 04 — Orquestração e Agentes Autônomos (10h)**
Implementação de Function Calling para integração com sistemas externos. Construção de agentes autônomos e fluxos de raciocínio com LangGraph e smolagents (consumindo o RAG e outras APIs).

**Módulo 05 — Produção, Observabilidade e Ferramental (4h)**
Monitoramento de telemetria, custos e rastreabilidade (Langfuse). Mitigação de riscos (Prompt Injection). Integração de assistentes de codificação (Claude Code, soluções open-source) ao fluxo de desenvolvimento diário.

> **📌 Foco prático:** A maior carga horária está no Módulo 04, que cobre o tema mais relevante e aplicado do mercado atual: agentes autônomos.

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

Acesse a pasta [`conteudo/`](./conteudo/) para o material teórico completo, cobrindo todos os módulos da ementa de forma progressiva.

## 🛠️ Práticas

Acesse a pasta [`praticas/`](./praticas/) para as atividades práticas evolutivas, que vão de uma simples chamada à API até a construção de agentes autônomos completos.

---

## 🗺️ Trilha de Aprendizado

```
[01 Fundamentos + Prompts] → [02 APIs + Estado]
                                      ↓
                              [03 RAG como Contexto]
                                      ↓
                         [04 Agentes Autônomos ★★★]
                                      ↓
                         [05 Produção + Ferramental]
```

> **★★★ Módulo com maior carga horária (10h)** — foco total em agentes autônomos, o diferencial do profissional de IA hoje.

**Cada módulo teórico é seguido de práticas correspondentes** para fixação e aplicação imediata do conteúdo.

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
