# Tópicos Avançados em TI — IA Generativa para Programadores

> **Disciplina:** Tópicos Avançados em TI  
> **Instituição:** IFPE — Instituto Federal de Pernambuco  
> **Professor:** Hélio Bentzen  
> **Carga Horária:** 28 horas  

---

## 🎯 Objetivo da Disciplina

Esta disciplina tem como objetivo capacitar programadores a **entender e construir soluções práticas com IA Generativa**, com foco no que é relevante para o mercado agora e nos próximos anos. Ao final do curso, o aluno será capaz de:

- Avaliar onde a IA generativa entra (e onde não entra) no desenho de um sistema
- Entender a arquitetura de LLMs, tokens, janela de contexto e seus limites
- Aplicar engenharia de contexto: system prompts, papéis, formatos de saída e decomposição de tarefas
- Gerenciar memória, estado e políticas do que entra e sai do contexto
- Implementar RAG com fontes externas, chunking, metadados e avaliação
- Construir agentes com contratos de ferramentas, integração com APIs e orquestração
- Monitorar aplicações de IA com observabilidade, traces, custos e regressão de comportamento

> **📌 Nota:** Esta disciplina é projetada para uso educacional. Todas as ferramentas e frameworks utilizados possuem opções **gratuitas ou open-source**. Modelos locais via [Ollama](https://ollama.ai) são utilizados como alternativa sem custo às APIs comerciais.

---

## 📋 Ementa

**IA Generativa para Desenvolvimento de Software — 28h**

| # | Módulo | Carga Horária |
|---|--------|---------------|
| 01 | [**IA Generativa no Desenho de Sistemas**](./conteudo/parte-01-ia-generativa-desenho-sistemas.md) | 2h |
| 02 | [**LLMs e Consumo de Contexto**](./conteudo/parte-02-llms-consumo-contexto.md) | 3h |
| 03 | [**Engenharia de Contexto I**](./conteudo/parte-03-engenharia-contexto-i.md) | 4h |
| 04 | [**Engenharia de Contexto II**](./conteudo/parte-04-engenharia-contexto-ii.md) | 3h |
| 05 | [**Conhecimento Externo e RAG**](./conteudo/parte-05-conhecimento-externo-rag.md) | 6h |
| 06 | [**Agentes no Sistema**](./conteudo/parte-06-agentes-no-sistema.md) | 7h |
| 07 | [**Observabilidade e Regressão de Comportamento**](./conteudo/parte-07-observabilidade-regressao.md) | 3h |
| **Total** | | **28h** |

### Detalhamento dos Módulos

**[Módulo 01 — IA Generativa no Desenho de Sistemas](./conteudo/parte-01-ia-generativa-desenho-sistemas.md) (2h)**
Fronteiras do produto, requisitos não funcionais (custo, latência, risco), onde o modelo entra e onde não entra.

**[Módulo 02 — LLMs e Consumo de Contexto](./conteudo/parte-02-llms-consumo-contexto.md) (3h)**
Tokens, janela de contexto, implicações para arquitetura e limites do modelo.

**[Módulo 03 — Engenharia de Contexto I](./conteudo/parte-03-engenharia-contexto-i.md) (4h)**
Instruções de sistema, papéis, formato de saída, decomposição em etapas; especificação testável.

**[Módulo 04 — Engenharia de Contexto II](./conteudo/parte-04-engenharia-contexto-ii.md) (3h)**
Memória, estado, políticas do que entra e sai do contexto.

**[Módulo 05 — Conhecimento Externo e RAG](./conteudo/parte-05-conhecimento-externo-rag.md) (6h)**
Fontes, modelagem da informação, chunking, metadados, atualização; avaliação básica.

**[Módulo 06 — Agentes no Sistema](./conteudo/parte-06-agentes-no-sistema.md) (7h)**
Contratos de ferramentas, integração com APIs e legados, orquestração, falhas e loops; eixo principal + panorama do ecossistema.

**[Módulo 07 — Observabilidade e Regressão de Comportamento](./conteudo/parte-07-observabilidade-regressao.md) (3h)**
Traces, custos, experimentos; degradação quando contexto ou dados mudam.

> **📌 Foco prático:** A maior carga horária está no Módulo 06 (Agentes), que cobre o tema mais relevante e aplicado do mercado atual, seguido do Módulo 05 (RAG), que fornece a base de conhecimento externo para os agentes.

---

## 📁 Estrutura do Repositório

```
ta-ti/
├── README.md                  ← Você está aqui
│
├── conteudo/                  ← Material teórico da disciplina
│   ├── parte-01-ia-generativa-desenho-sistemas.md
│   ├── parte-02-llms-consumo-contexto.md
│   ├── parte-03-engenharia-contexto-i.md
│   ├── parte-04-engenharia-contexto-ii.md
│   ├── parte-05-conhecimento-externo-rag.md
│   ├── parte-06-agentes-no-sistema.md
│   └── parte-07-observabilidade-regressao.md
│
└── praticas/                  ← Atividades práticas evolutivas
    ├── pratica-01-primeiros-passos-llm.md
    ├── pratica-02-engenharia-contexto.md
    ├── pratica-03-conhecimento-externo-rag.md
    ├── pratica-04-agente-simples.md
    ├── pratica-05-agente-ferramentas.md
    └── pratica-06-observabilidade-projeto-final.md
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

Material teórico completo, cobrindo todos os módulos da ementa de forma progressiva:

1. [Parte 01 — IA Generativa no Desenho de Sistemas](./conteudo/parte-01-ia-generativa-desenho-sistemas.md)
2. [Parte 02 — LLMs e Consumo de Contexto](./conteudo/parte-02-llms-consumo-contexto.md)
3. [Parte 03 — Engenharia de Contexto I](./conteudo/parte-03-engenharia-contexto-i.md)
4. [Parte 04 — Engenharia de Contexto II](./conteudo/parte-04-engenharia-contexto-ii.md)
5. [Parte 05 — Conhecimento Externo e RAG](./conteudo/parte-05-conhecimento-externo-rag.md)
6. [Parte 06 — Agentes no Sistema](./conteudo/parte-06-agentes-no-sistema.md)
7. [Parte 07 — Observabilidade e Regressão de Comportamento](./conteudo/parte-07-observabilidade-regressao.md)

## 🛠️ Práticas

Atividades práticas evolutivas, que vão de uma simples chamada à API até a construção de agentes autônomos completos:

1. [Prática 01 — Primeiros Passos com LLM](./praticas/pratica-01-primeiros-passos-llm.md)
2. [Prática 02 — Engenharia de Contexto](./praticas/pratica-02-engenharia-contexto.md)
3. [Prática 03 — Conhecimento Externo e RAG](./praticas/pratica-03-conhecimento-externo-rag.md)
4. [Prática 04 — Agente Simples](./praticas/pratica-04-agente-simples.md)
5. [Prática 05 — Agente com Ferramentas](./praticas/pratica-05-agente-ferramentas.md)
6. [Prática 06 — Observabilidade e Projeto Final](./praticas/pratica-06-observabilidade-projeto-final.md)

---

## 🗺️ Trilha de Aprendizado

```
[01 IA no Desenho de Sistemas] → [02 LLMs e Contexto]
                                          ↓
                              [03 Engenharia de Contexto I]
                                          ↓
                              [04 Engenharia de Contexto II]
                                          ↓
                              [05 Conhecimento Externo e RAG ★★]
                                          ↓
                              [06 Agentes no Sistema ★★★]
                                          ↓
                              [07 Observabilidade e Regressão]
```

> **★★★ Módulo com maior carga horária (7h)** — foco em agentes no sistema, o diferencial do profissional de IA hoje.
> **★★ Segundo maior módulo (6h)** — RAG é a base de conhecimento que alimenta os agentes.

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
