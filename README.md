# Tópicos Avançados em TI — IA Generativa para Programadores

> **Disciplina:** Tópicos Avançados em TI  
> **Instituição:** IFPE — Instituto Federal de Pernambuco  
> **Professor:** Hélio Bentzen  
> **Carga Horária:** 27 horas  

---

## 🎯 Objetivo da Disciplina

Esta disciplina capacita programadores a **projetar e construir sistemas com IA Generativa** com foco no que é realmente usado no mercado — não apenas em chamar APIs, mas em tomar decisões de arquitetura, gerenciar contexto de forma eficiente, integrar conhecimento externo, orquestrar agentes e medir o comportamento do sistema em produção.

Ao final do curso, o aluno será capaz de:

- Decidir **onde** e **quando** um LLM faz sentido em um sistema — e quando não faz
- Gerenciar tokens, janela de contexto e custos como restrições de engenharia reais
- Escrever instruções de sistema testáveis e especificações de saída estruturada
- Implementar estratégias de memória e estado para conversas multi-turno
- Construir pipelines RAG com chunking, metadados, retrieval e avaliação
- Desenvolver agentes com contratos de ferramentas claros e tratamento de falhas
- Monitorar aplicações de IA com traces, métricas de custo e detecção de regressão

> **📌 Abordagem crítica:** O curso analisa o que é hype versus o que é amplamente adotado na indústria. Cada módulo discute trade-offs reais e limitações práticas.

---

## 📋 Ementa

| # | Tema | Carga Horária |
|---|------|---------------|
| 01 | IA Generativa no Desenho de Sistemas | 2h |
| 02 | LLMs e Consumo de Contexto | 3h |
| 03 | Engenharia de Contexto I — Instruções, Papéis e Saída Estruturada | 4h |
| 04 | Engenharia de Contexto II — Memória, Estado e Políticas | 3h |
| 05 | Conhecimento Externo e RAG | 6h |
| 06 | **Agentes no Sistema** | **7h** |
| 07 | Observabilidade e Regressão de Comportamento | 2h |
| **Total** | | **27h** |

> **📌 Foco prático:** A carga horária está concentrada nos módulos 05 e 06, que cobrem os temas de maior demanda no mercado. O módulo 07 fecha o ciclo com o que diferencia sistemas de IA em produção de protótipos.

---

## 📁 Estrutura do Repositório

```
ta-ti/
├── README.md                          ← Você está aqui
│
├── conteudo/                          ← Material teórico da disciplina
│   ├── parte-01-ia-generativa-desenho-sistemas.md
│   ├── parte-02-llms-consumo-contexto.md
│   ├── parte-03-engenharia-de-contexto-1.md
│   ├── parte-04-engenharia-de-contexto-2.md
│   ├── parte-05-conhecimento-externo-rag.md
│   ├── parte-06-agentes-no-sistema.md
│   └── parte-07-observabilidade-regressao.md
│
└── praticas/                          ← Atividades práticas
    ├── pratica-01-desenho-de-sistemas.md
    ├── pratica-02-llms-e-contexto.md
    ├── pratica-03-engenharia-de-contexto-1.md
    ├── pratica-04-engenharia-de-contexto-2.md
    ├── pratica-05-rag.md
    ├── pratica-06-agentes.md
    └── pratica-07-observabilidade.md
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
pip install openai python-dotenv tiktoken

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

> **💡 Dica:** A maioria dos exercícios pode ser realizada usando apenas o **Ollama** com modelos open-source, sem necessidade de API keys pagas.

---

## 📚 Conteúdo

Acesse a pasta [`conteudo/`](./conteudo/) para o material teórico completo, dividido em 7 partes progressivas.

## 🛠️ Práticas

Acesse a pasta [`praticas/`](./praticas/) para as 7 atividades práticas, cada uma correspondendo a uma parte teórica.

---

## 🗺️ Trilha de Aprendizado

```
[01 Desenho de Sistemas] → [02 LLMs e Contexto]
                                    ↓
              [03 Eng. Contexto I] → [04 Eng. Contexto II]
                                    ↓
                       [05 Conhecimento Externo e RAG]
                                    ↓
                         [06 Agentes no Sistema ★]
                                    ↓
                  [07 Observabilidade e Regressão]
```

> **★ Módulo com maior carga horária** — foco em integração, orquestração e casos reais.

**Cada parte teórica é seguida de uma prática correspondente** para fixação e aplicação imediata do conteúdo.

---

## 💡 Metodologia

- **Aulas expositivas curtas** com foco em conceitos essenciais e trade-offs reais
- **Análise crítica** do ecossistema: o que usar, quando e por quê
- **Práticas hands-on** com problemas progressivos
- **Código funcional** que pode ser adaptado para projetos reais

---

## 📬 Contato

- **Professor:** Hélio Bentzen
- **Instituição:** IFPE — Campus Recife
- **Disciplina:** Tópicos Avançados em TI

---

> *"A melhor forma de entender IA é construindo com ela."*
