# Parte 01 — IA Generativa no Desenho de Sistemas

> **Carga horária:** 2 horas  
> **Prática correspondente:** [Prática 01](../praticas/pratica-01-primeiros-passos-llm.md)

---

## 1.1 O que é Inteligência Artificial Generativa?

IA Generativa é uma categoria de modelos de inteligência artificial capazes de **criar conteúdo novo** — texto, imagens, código, áudio, vídeo — a partir de padrões aprendidos em grandes volumes de dados.

Diferente dos sistemas de IA tradicionais, que classificam ou preveem com base em regras e padrões pré-definidos, os modelos generativos aprendem a **distribuição dos dados** de treinamento e conseguem amostrar novos dados dessa distribuição.

### Tipos de IA Generativa

| Tipo | O que gera | Exemplos |
|------|-----------|---------|
| LLMs | Texto, código | GPT-4, Claude, Llama |
| Difusão | Imagens | DALL-E, Stable Diffusion, Midjourney |
| Multimodal | Texto + imagem + áudio | Gemini, GPT-4o |
| Áudio/Música | Sons, fala, música | Whisper, Suno, ElevenLabs |
| Vídeo | Clipes de vídeo | Sora, Runway |

---

## 1.2 Breve Histórico

```
1950s → Alan Turing propõe o "Teste de Turing"
1980s → Redes neurais simples (perceptrons)
2012  → AlexNet vence ImageNet (deep learning explode)
2017  → Artigo "Attention Is All You Need" (arquitetura Transformer)
2018  → BERT (Google) — modelos pré-treinados de linguagem
2019  → GPT-2 (OpenAI) — geração de texto impressiona o mundo
2020  → GPT-3 — 175 bilhões de parâmetros
2022  → ChatGPT lançado — adoção de massa em 5 dias
2023  → GPT-4, Claude, Llama, Gemini — corrida dos modelos
2024+ → Agentes, multimodalidade, modelos menores e mais eficientes
```

O marco principal foi a publicação do artigo **"Attention Is All You Need"** (Vaswani et al., 2017), que introduziu a arquitetura **Transformer** — base de praticamente todos os LLMs modernos.

---

## 1.3 Por que Agora?

Três fatores convergiram para tornar a IA Generativa viável e poderosa:

1. **Dados em escala**: a internet forneceu trilhões de tokens de texto
2. **Hardware (GPUs/TPUs)**: poder de computação paralela em larga escala
3. **Algoritmos**: a arquitetura Transformer e técnicas de treinamento eficientes

---

## 1.4 O Impacto para Programadores

Como programador, você é afetado em duas dimensões:

### Dimensão 1 — Produtividade

Ferramentas como GitHub Copilot, Cursor e ChatGPT já mudaram o desenvolvimento:
- Autocompletar código inteligente
- Geração de testes automaticamente
- Explicação e refatoração de código legado
- Documentação automática

### Dimensão 2 — Novas Aplicações

Você pode agora **construir produtos e ferramentas** que antes eram impossíveis:
- Chatbots com contexto de negócio
- Sistemas de busca semântica
- Automação de processos com linguagem natural
- Assistentes especializados (jurídico, médico, técnico)
- Agentes autônomos que executam tarefas complexas

---

## 1.5 Conceitos Fundamentais

### Token

A unidade básica de processamento de um LLM. Um token ≈ 4 caracteres em inglês / ≈ 3 caracteres em português.

```
"Olá, mundo!" → ["Ol", "á", ",", " mundo", "!"]  ← exemplo ilustrativo
```

**Por que importa?** Modelos têm um limite de tokens (janela de contexto) e cobram por token usado.

### Contexto (Context Window)

A quantidade máxima de tokens que o modelo pode "ver" de uma vez. É a "memória de trabalho" do modelo.

| Modelo | Contexto |
|--------|----------|
| GPT-3.5 | 16k tokens |
| GPT-4o | 128k tokens |
| Claude 3.5 Sonnet | 200k tokens |
| Gemini 1.5 Pro | 1M tokens |

### Temperatura

Parâmetro que controla a **aleatoriedade** das respostas:
- `temperatura = 0.0` → respostas determinísticas e conservadoras
- `temperatura = 1.0` → respostas mais criativas e variadas
- `temperatura > 1.0` → respostas imprevisíveis (geralmente indesejado)

### Parâmetros de um Modelo

São os "pesos" da rede neural — valores ajustados durante o treinamento. Mais parâmetros ≠ sempre melhor, mas geralmente indica maior capacidade.

---

## 1.6 O Ecossistema Atual

### Provedores de Modelos

| Provedor | Modelos | Destaques |
|----------|---------|-----------|
| OpenAI | GPT-4o, GPT-4.1, o1 | Pioneiros, API mais madura |
| Anthropic | Claude 3.5/3.7 | Foco em segurança e raciocínio |
| Google | Gemini 1.5/2.0 | Integração com Google Cloud |
| Meta | Llama 3.x | Open-source, uso local |
| Mistral | Mistral, Mixtral | Europeus, eficientes |
| Groq | Llama via Groq | Inferência ultra-rápida |

### Ferramentas para Desenvolvedores

- **LangChain / LangGraph**: orquestração de LLMs e agentes
- **LlamaIndex**: frameworks para RAG
- **ChromaDB / Pinecone / Weaviate**: bancos de dados vetoriais
- **Hugging Face**: modelos open-source e datasets
- **Ollama**: executar modelos localmente

---

## 1.7 Limitações Importantes

Conhecer as limitações é tão importante quanto conhecer as capacidades:

| Limitação | Descrição |
|-----------|-----------|
| **Alucinações** | Modelos inventam fatos com confiança |
| **Corte de conhecimento** | Dados de treinamento têm data de corte |
| **Contexto limitado** | Não lembram interações passadas por padrão |
| **Custo** | APIs comerciais têm custo por token |
| **Latência** | Geração de tokens leva tempo |
| **Consistência** | Respostas podem variar entre execuções |
| **Viés** | Refletem vieses dos dados de treinamento |

---

## 1.8 Ética e Responsabilidade

Como construtor de ferramentas com IA, você tem responsabilidade sobre:

- **Transparência**: deixar claro quando o sistema usa IA
- **Privacidade**: não enviar dados sensíveis a APIs externas sem consentimento
- **Confiabilidade**: validar saídas de modelos antes de usá-las em produção
- **Segurança**: proteger contra prompt injection e uso malicioso
- **Acessibilidade**: garantir que a ferramenta funcione para todos os usuários

---

## 📌 Resumo da Parte 01

| Conceito | Definição |
|----------|-----------|
| IA Generativa | IA que cria novo conteúdo a partir de padrões aprendidos |
| Transformer | Arquitetura base dos LLMs modernos (2017) |
| Token | Unidade básica de texto processada pelo modelo |
| Contexto | "Memória de trabalho" do modelo |
| Temperatura | Controla aleatoriedade da geração |
| Alucinação | Geração de informações falsas com aparência de verdadeiras |

---

## 🔗 Referências e Leitura Adicional

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — artigo original do Transformer
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — explicação visual
- [OpenAI Cookbook](https://cookbook.openai.com) — exemplos práticos
- [Hugging Face Course](https://huggingface.co/learn) — curso gratuito sobre NLP/LLMs

---

➡️ **Próximo:** [Parte 02 — LLMs e Consumo de Contexto](./parte-02-llms-consumo-contexto.md)
🏠 **Início:** [README](../README.md)
