# Tópicos Avançados em TI — IA Generativa para Programadores

> **Disciplina:** Tópicos Avançados em TI  
> **Instituição:** IFPE — Instituto Federal de Pernambuco  
> **Professor:** Hélio Bentzen  
> **Carga Horária:** 28 horas  

---

## 🎯 Objetivo da Disciplina

Esta disciplina tem como objetivo capacitar programadores a **entender e construir soluções práticas com IA Generativa**, com foco no que é relevante para o mercado agora e nos próximos anos. O conteúdo parte do zero — explicando o que são LLMs e como funcionam — e avança progressivamente até a construção de agentes autônomos com ferramentas e observabilidade. Ao final do curso, o aluno será capaz de:

- **Avaliar onde a IA generativa se encaixa** (e onde não se encaixa) no desenho de um sistema, considerando custo, latência e riscos
- **Entender a arquitetura de LLMs** — como tokens, embeddings, atenção e janela de contexto afetam o que o modelo pode e não pode fazer
- **Aplicar engenharia de contexto** — estruturar system prompts, atribuir papéis, definir formatos de saída e decompor tarefas complexas em etapas
- **Gerenciar memória e estado** — controlar o que entra e sai da janela de contexto entre chamadas ao modelo
- **Implementar RAG (Retrieval Augmented Generation)** — conectar LLMs a fontes externas de conhecimento usando embeddings, chunking e bancos vetoriais
- **Construir agentes de IA** — criar sistemas que raciocinam, usam ferramentas, interagem com APIs e orquestram tarefas complexas
- **Monitorar aplicações de IA** — implementar observabilidade com traces, controlar custos e detectar regressão de comportamento

> **📌 Nota:** Esta disciplina é projetada para uso educacional. Todas as ferramentas e frameworks utilizados possuem opções **gratuitas ou open-source**. Modelos locais via [Ollama](https://ollama.ai) são utilizados como alternativa sem custo às APIs comerciais.

---

## 📋 Ementa

**IA Generativa para Desenvolvimento de Software — 28h**

| # | Módulo | Foco Principal | Carga Horária |
|---|--------|----------------|---------------|
| 01 | [**IA Generativa no Desenho de Sistemas**](./conteudo/parte-01-ia-generativa-desenho-sistemas.md) | Visão geral, conceitos fundamentais e onde a IA se encaixa | 2h |
| 02 | [**LLMs e Consumo de Contexto**](./conteudo/parte-02-llms-consumo-contexto.md) | Como LLMs funcionam por dentro: Transformers, tokens e APIs | 3h |
| 03 | [**Engenharia de Contexto I**](./conteudo/parte-03-engenharia-contexto-i.md) | Técnicas de prompt: zero/few-shot, CoT e saída estruturada | 4h |
| 04 | [**Engenharia de Contexto II**](./conteudo/parte-04-engenharia-contexto-ii.md) | Memória, estado e políticas do que entra no contexto | 3h |
| 05 | [**Conhecimento Externo e RAG**](./conteudo/parte-05-conhecimento-externo-rag.md) | Embeddings, bancos vetoriais e retrieval augmented generation | 6h |
| 06 | [**Agentes no Sistema**](./conteudo/parte-06-agentes-no-sistema.md) | Agentes com ferramentas, APIs, orquestração e loops | 7h |
| 07 | [**Observabilidade e Regressão de Comportamento**](./conteudo/parte-07-observabilidade-regressao.md) | Traces, custos, métricas e detecção de degradação | 3h |
| **Total** | | | **28h** |

### Detalhamento dos Módulos

**[Módulo 01 — IA Generativa no Desenho de Sistemas](./conteudo/parte-01-ia-generativa-desenho-sistemas.md) (2h)**
O que é IA generativa, breve histórico, conceitos fundamentais (tokens, contexto, temperatura) e o ecossistema atual de modelos e ferramentas. Discussão sobre onde o modelo se encaixa no sistema — e onde não se encaixa — considerando fronteiras do produto, requisitos não funcionais (custo, latência, risco) e limitações da tecnologia.

**[Módulo 02 — LLMs e Consumo de Contexto](./conteudo/parte-02-llms-consumo-contexto.md) (3h)**
Como um LLM funciona por dentro: a arquitetura Transformer, mecanismo de atenção, tokenização, embeddings, pré-treinamento e fine-tuning (RLHF/DPO). Exploração prática com Ollama (modelos locais gratuitos), uso de APIs (OpenAI e Anthropic), streaming, gerenciamento de histórico e estratégias de decodificação (temperatura, top-p, top-k).

**[Módulo 03 — Engenharia de Contexto I](./conteudo/parte-03-engenharia-contexto-i.md) (4h)**
A arte de estruturar prompts para obter respostas precisas e consistentes. Cobre a anatomia de um bom prompt, técnicas essenciais (zero-shot, few-shot, chain-of-thought, ReAct), técnicas avançadas (role prompting, saída estruturada, prompt chaining), gestão de tokens e janela de contexto, segurança contra prompt injection, e como avaliar a qualidade de prompts.

**[Módulo 04 — Engenharia de Contexto II](./conteudo/parte-04-engenharia-contexto-ii.md) (3h)**
Gestão de memória e estado em aplicações com LLMs. Como LLMs são stateless por natureza, o desenvolvedor precisa controlar o que entra e sai da janela de contexto. Aborda padrões de gerenciamento de histórico (janela deslizante, resumo progressivo), políticas de contexto, memória de longo prazo, estratégias de posicionamento de informação no prompt, filtragem de segurança e debugging de contexto.

**[Módulo 05 — Conhecimento Externo e RAG](./conteudo/parte-05-conhecimento-externo-rag.md) (6h)**
Como conectar LLMs a bases de conhecimento externas usando RAG (Retrieval Augmented Generation). Parte de embeddings e busca semântica, passa por bancos de dados vetoriais (ChromaDB), técnicas de chunking e modelagem da informação, até a implementação completa de um pipeline RAG. Inclui avaliação básica de qualidade de respostas e estratégias de atualização de dados.

**[Módulo 06 — Agentes no Sistema](./conteudo/parte-06-agentes-no-sistema.md) (7h)**
O módulo mais extenso e aplicado do curso. Cobre o que é um agente de IA, o loop ReAct (Reason + Act), contratos de ferramentas (function calling), integração com APIs e sistemas legados, orquestração de múltiplas ferramentas, tratamento de falhas e loops infinitos. Inclui um panorama do ecossistema de frameworks (LangGraph, CrewAI, etc.) e padrões de segurança para agentes em produção.

**[Módulo 07 — Observabilidade e Regressão de Comportamento](./conteudo/parte-07-observabilidade-regressao.md) (3h)**
Como monitorar e manter aplicações com IA em produção. Cobre padrões de arquitetura para ferramentas com LLM, implementação de traces para depuração, controle de custos por chamada, métricas de qualidade, experimentos A/B com prompts, e como detectar e tratar degradação de comportamento quando o contexto ou os dados mudam ao longo do tempo.

> **📌 Foco prático:** A maior carga horária está no Módulo 06 — Agentes (7h), que cobre o tema mais relevante e aplicado do mercado atual: sistemas que raciocinam e agem usando ferramentas. O Módulo 05 — RAG (6h) vem em seguida, fornecendo a base de conhecimento externo que alimenta esses agentes. Os módulos anteriores constroem os fundamentos necessários de forma progressiva.

---

## 📁 Estrutura do Repositório

O repositório é organizado em duas pastas principais: **conteudo/** contém o material teórico completo de cada módulo, e **praticas/** contém as atividades hands-on correspondentes. Cada prática referencia o módulo teórico que a fundamenta.

```
ta-ti/
├── README.md                  ← Você está aqui — visão geral do curso
│
├── conteudo/                  ← Material teórico da disciplina (7 módulos)
│   ├── parte-01-ia-generativa-desenho-sistemas.md   ← Conceitos e ecossistema
│   ├── parte-02-llms-consumo-contexto.md            ← Transformers, tokens, APIs
│   ├── parte-03-engenharia-contexto-i.md            ← Técnicas de prompt
│   ├── parte-04-engenharia-contexto-ii.md           ← Memória e estado
│   ├── parte-05-conhecimento-externo-rag.md         ← Embeddings e RAG
│   ├── parte-06-agentes-no-sistema.md               ← Agentes e ferramentas
│   └── parte-07-observabilidade-regressao.md        ← Monitoramento e métricas
│
└── praticas/                  ← Atividades práticas evolutivas (6 práticas)
    ├── pratica-01-primeiros-passos-llm.md           ← Primeira chamada à API e chatbot
    ├── pratica-02-engenharia-contexto.md            ← Prompts, CoT e extração de dados
    ├── pratica-03-conhecimento-externo-rag.md       ← Pipeline RAG completo
    ├── pratica-04-agente-simples.md                 ← Primeiro agente com ferramentas
    ├── pratica-05-agente-ferramentas.md             ← Agente multi-ferramentas e RAG
    └── pratica-06-observabilidade-projeto-final.md  ← Observabilidade e projeto integrador
```

---

## 🚀 Como Usar Este Repositório

Este repositório contém todo o material do curso. Comece lendo os módulos teóricos na pasta `conteudo/` na ordem numérica e, ao final de cada bloco, faça a prática correspondente na pasta `praticas/`. Os exercícios são cumulativos — cada prática usa conceitos e código das anteriores.

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

Material teórico completo, cobrindo todos os módulos da ementa de forma progressiva — do básico ao avançado. Cada parte pode ser lida de forma independente, mas seguem uma sequência lógica onde conceitos anteriores fundamentam os posteriores:

1. [Parte 01 — IA Generativa no Desenho de Sistemas](./conteudo/parte-01-ia-generativa-desenho-sistemas.md) — O que é IA generativa, por que importa agora, conceitos fundamentais (token, contexto, temperatura), ecossistema de modelos e ferramentas, limitações e ética
2. [Parte 02 — LLMs e Consumo de Contexto](./conteudo/parte-02-llms-consumo-contexto.md) — Arquitetura Transformer, atenção, pré-treinamento, RLHF, geração de texto, uso de Ollama (local e gratuito) e APIs de LLMs
3. [Parte 03 — Engenharia de Contexto I](./conteudo/parte-03-engenharia-contexto-i.md) — Anatomia do prompt, zero-shot, few-shot, chain-of-thought, saída estruturada, prompt chaining, gestão de tokens e segurança
4. [Parte 04 — Engenharia de Contexto II](./conteudo/parte-04-engenharia-contexto-ii.md) — Memória e estado em apps com LLMs, padrões de histórico, políticas de contexto, memória de longo prazo e filtragem
5. [Parte 05 — Conhecimento Externo e RAG](./conteudo/parte-05-conhecimento-externo-rag.md) — Embeddings, busca semântica, bancos vetoriais (ChromaDB), chunking, pipeline RAG completo e avaliação de qualidade
6. [Parte 06 — Agentes no Sistema](./conteudo/parte-06-agentes-no-sistema.md) — Agentes de IA, loop ReAct, function calling, integração com APIs, orquestração, tratamento de falhas e panorama de frameworks
7. [Parte 07 — Observabilidade e Regressão de Comportamento](./conteudo/parte-07-observabilidade-regressao.md) — Padrões de arquitetura para IA em produção, traces, custos, métricas e detecção de degradação

## 🛠️ Práticas

Atividades práticas evolutivas, que vão de uma simples chamada à API até a construção de agentes autônomos completos. Cada prática inclui exercícios guiados passo a passo, desafios opcionais para aprofundamento e um checklist de entrega:

1. [Prática 01 — Primeiros Passos com LLM](./praticas/pratica-01-primeiros-passos-llm.md) — Configuração do ambiente, primeira chamada à API, exploração de parâmetros (temperatura), system prompts com personas e chatbot com histórico
2. [Prática 02 — Engenharia de Contexto](./praticas/pratica-02-engenharia-contexto.md) — Comparação zero-shot vs few-shot, chain-of-thought na prática, extração de dados estruturados (JSON) e sistema de triagem de tickets
3. [Prática 03 — Conhecimento Externo e RAG](./praticas/pratica-03-conhecimento-externo-rag.md) — Geração de embeddings, busca por similaridade, construção de pipeline RAG completo com ChromaDB e avaliação de respostas
4. [Prática 04 — Agente Simples](./praticas/pratica-04-agente-simples.md) — Primeiro agente com function calling, loop de raciocínio, uso de ferramentas e integração básica
5. [Prática 05 — Agente com Ferramentas](./praticas/pratica-05-agente-ferramentas.md) — Agente com múltiplas ferramentas especializadas, RAG como ferramenta do agente e raciocínio multi-step
6. [Prática 06 — Observabilidade e Projeto Final](./praticas/pratica-06-observabilidade-projeto-final.md) — Implementação de traces e métricas, controle de custos e projeto integrador que une todos os conceitos do curso

---

## 🗺️ Trilha de Aprendizado

O curso segue uma progressão linear, onde cada módulo constrói sobre os anteriores. Os módulos iniciais estabelecem os fundamentos (o que é IA generativa, como LLMs funcionam), os módulos intermediários ensinam a controlar o modelo (engenharia de contexto) e a conectá-lo a dados externos (RAG), e os módulos finais focam na construção de agentes autônomos e na manutenção de sistemas em produção.

```
[01 IA no Desenho de Sistemas] → [02 LLMs e Contexto]
        Fundamentos                  Arquitetura
                                          ↓
                              [03 Engenharia de Contexto I]
                                   Técnicas de prompt
                                          ↓
                              [04 Engenharia de Contexto II]
                                   Memória e estado
                                          ↓
                              [05 Conhecimento Externo e RAG ★★]
                                Embeddings e busca semântica
                                          ↓
                              [06 Agentes no Sistema ★★★]
                              Ferramentas e orquestração
                                          ↓
                              [07 Observabilidade e Regressão]
                                Monitoramento e métricas
```

> **★★★ Módulo com maior carga horária (7h)** — foco em agentes no sistema: o profissional que sabe construir agentes com ferramentas é o diferencial do mercado de IA hoje.
> **★★ Segundo maior módulo (6h)** — RAG é a técnica que permite ao LLM acessar dados atualizados e específicos do negócio, sendo a base de conhecimento que alimenta os agentes.

**Cada módulo teórico é acompanhado de práticas correspondentes** para fixação e aplicação imediata do conteúdo. A ideia é que o aluno nunca passe mais de uma aula sem colocar a mão no código.

---

## 💡 Metodologia

O curso combina teoria e prática em ciclos curtos, priorizando a construção de intuição através da experimentação:

- **Aulas expositivas curtas** — conceitos essenciais apresentados de forma direta, sem excesso de teoria. Cada aula foca em um tema específico e termina com conexão ao próximo
- **Demonstrações ao vivo** — código real executado em sala, mostrando como as APIs funcionam, como os modelos se comportam e onde as coisas podem dar errado
- **Práticas hands-on** — exercícios progressivos que vão do simples (uma chamada à API) ao complexo (agente autônomo com múltiplas ferramentas). O aluno constrói em cima do que já fez
- **Projeto final integrado** — une todos os conceitos em uma ferramenta funcional com IA: agente com RAG, ferramentas e observabilidade

---

## 📬 Contato

- **Professor:** Hélio Bentzen
- **Instituição:** IFPE — Campus Recife
- **Disciplina:** Tópicos Avançados em TI

---

> *"A melhor forma de entender IA é construindo com ela."*
