# Parte 02 — LLMs: Como Funcionam os Grandes Modelos de Linguagem

> **Carga horária:** 3 horas  
> **Prática correspondente:** [Prática 01](../praticas/pratica-01-primeiros-passos-llm.md)

---

## 2.1 O que é um LLM?

Um **Large Language Model (LLM)** é uma rede neural de escala massiva treinada para prever o próximo token em uma sequência de texto. Apesar da simplicidade do objetivo de treinamento, ao fazê-lo em escala suficiente com dados suficientes, emergem capacidades surpreendentes.

> **Intuição central:** Um LLM é, na essência, um modelo probabilístico de linguagem. Dado um contexto, ele calcula a distribuição de probabilidade sobre o vocabulário e escolhe o próximo token.

---

## 2.2 A Arquitetura Transformer

A arquitetura **Transformer**, publicada em 2017, é a base de todos os LLMs modernos. Entender seus componentes principais ajuda a compreender as capacidades e limitações dos modelos.

### Componentes Principais

```
Entrada (tokens) → Embedding → [Bloco Transformer × N] → Saída (logits)
```

#### 1. Tokenização

Antes de qualquer processamento, o texto é convertido em tokens usando um **tokenizador**:

```python
# Exemplo com tiktoken (tokenizador da OpenAI)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Olá, mundo da IA!")
print(tokens)        # [3838, 11, 1629, 2892, 50218, 0]  (exemplo)
print(len(tokens))   # 6
```

#### 2. Embedding

Cada token é convertido em um vetor de alta dimensão (ex: 768, 1024, 4096 dimensões). Esses vetores carregam o significado semântico do token.

#### 3. Mecanismo de Atenção (Self-Attention)

O coração do Transformer. Permite que cada token "olhe" para todos os outros tokens no contexto e calcule o quanto cada um é relevante para ele.

```
Q (Query) × K (Key) → Pesos de atenção → × V (Value) = Saída
```

**Intuição**: ao processar a palavra "banco" em "fui ao banco de areia", a atenção conecta "banco" a "areia", desambiguando o sentido.

#### 4. Multi-Head Attention

Múltiplas "cabeças" de atenção rodam em paralelo, cada uma capturando diferentes tipos de relação entre tokens (sintática, semântica, coreferencialidade, etc.).

#### 5. Feed-Forward Network

Após a atenção, cada token passa por uma rede MLP (multilayer perceptron) que transforma sua representação. É onde grande parte do "conhecimento factual" é armazenado.

#### 6. Normalização e Conexões Residuais

Técnicas de estabilização numérica que permitem treinar redes muito profundas (centenas de camadas).

---

## 2.3 Pré-treinamento

O pré-treinamento é a fase mais cara e impactante. O modelo aprende a prever o próximo token em bilhões/trilhões de tokens de texto da internet.

### Objetivo de Treinamento

```
Maximize P(token_n | token_1, token_2, ..., token_{n-1})
```

### Dados de Treinamento

- Web crawls (CommonCrawl, C4)
- Livros digitalizados (Books3, Project Gutenberg)
- Código-fonte (GitHub)
- Artigos científicos (arXiv)
- Enciclopédias (Wikipedia)
- E muito mais...

### Escala e Custo

| Modelo | Parâmetros | Tokens de treino | Custo estimado |
|--------|-----------|-----------------|----------------|
| GPT-3 | 175B | 300B | ~$4.6M |
| GPT-4 | ~1T (estimado) | ~13T | ~$100M |
| Llama 3 70B | 70B | 15T | ~$10M |
| Llama 3 8B | 8B | 15T | ~$1M |

---

## 2.4 Fine-tuning e RLHF

Após o pré-treinamento, o modelo "sabe" muito sobre linguagem mas não necessariamente segue instruções de forma útil. O **fine-tuning** ajusta o modelo para comportamentos desejados.

### Instruction Tuning (SFT)

Treina o modelo em pares de (instrução, resposta desejada):

```
Instrução: "Resuma este texto em 3 pontos."
Resposta:  "1. ... 2. ... 3. ..."
```

### RLHF — Reinforcement Learning from Human Feedback

1. **Coleta de comparações**: humanos escolhem qual de duas respostas é melhor
2. **Modelo de recompensa**: treinado para prever preferências humanas
3. **RL**: o LLM é otimizado para maximizar a recompensa do modelo de preferência

```
LLM → gera resposta A e B
Humano → prefere A
Reward model → aprende que A > B
PPO/GRPO → ajusta LLM para gerar mais como A
```

### DPO — Direct Preference Optimization

Uma alternativa mais simples ao RLHF que otimiza diretamente as preferências sem um modelo de recompensa separado.

---

## 2.5 Geração de Texto

A geração acontece de forma **autorregressiva**: um token de cada vez, onde cada token gerado entra como contexto para o próximo.

### Estratégias de Decodificação

#### Greedy Decoding (temperatura = 0)
```python
# Sempre escolhe o token mais provável
token_next = argmax(logits)
```

#### Amostragem com Temperatura
```python
# Divide os logits pela temperatura antes do softmax
probs = softmax(logits / temperature)
token_next = sample(probs)
```

#### Top-p (Nucleus Sampling)
```python
# Considera apenas os tokens que somam probabilidade p
# Exemplo: top_p=0.9 usa os tokens que cobrem 90% da probabilidade
```

#### Top-k
```python
# Considera apenas os k tokens mais prováveis
# Exemplo: top_k=50
```

---

## 2.6 Janela de Contexto e KV Cache

### Janela de Contexto

O número máximo de tokens que o modelo processa de uma vez. É determinado durante o treinamento e não pode ser expandido facilmente.

- Tokens de entrada (prompt) + tokens de saída (resposta) devem caber na janela
- Quanto maior a janela, maior o custo de computação (quadrático na atenção)

### KV Cache

Durante a geração, as matrizes Key e Value dos tokens já processados são cacheadas para evitar recomputação. Isso acelera significativamente a geração, mas consome memória proporcional ao contexto.

---

## 2.7 Modelos e Suas Arquiteturas

Existem variantes da arquitetura Transformer:

| Tipo | Exemplo | Uso |
|------|---------|-----|
| Decoder-only | GPT, Llama, Claude | Geração de texto (causa-efeito) |
| Encoder-only | BERT, RoBERTa | Classificação, embeddings |
| Encoder-Decoder | T5, BART | Tradução, sumarização |

Os LLMs de chat (GPT, Claude, Llama) são quase todos **decoder-only**.

---

## 2.8 Por que LLMs São Tão Capazes?

Capacidades emergentes que surgem com escala:

- **In-context learning**: aprender com exemplos no prompt sem atualizar pesos
- **Chain-of-thought**: raciocinar passo a passo
- **Code generation**: gerar e depurar código
- **Instruction following**: seguir instruções complexas e multi-etapas
- **Multilinguismo**: funcionar em dezenas de idiomas sem treino específico

---

## 2.9 Limitações Técnicas

| Limitação | Causa Técnica |
|-----------|--------------|
| Alucinações | Otimização probabilística, não busca de verdade |
| Contexto fixo | Custo quadrático da atenção |
| Sem memória persistente | Pesos fixos após treinamento |
| Dificuldade com números | Tokenização e aritmética não são naturais |
| Viés | Dados de treinamento refletem vieses humanos |

---

## 📌 Resumo da Parte 02

| Conceito | Descrição |
|----------|-----------|
| Transformer | Arquitetura baseada em atenção — base de todos os LLMs |
| Self-Attention | Mecanismo que relaciona tokens entre si no contexto |
| Pré-treinamento | Aprendizado em escala: prever próximo token |
| RLHF | Alinhamento com preferências humanas via reforço |
| Temperatura | Controla aleatoriedade: 0 = determinístico |
| KV Cache | Otimização que evita recomputar atenção em tokens passados |

---

## 🔗 Referências

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Training language models to follow instructions with human feedback (RLHF)](https://arxiv.org/abs/2203.02155)

---

⬅️ **Anterior:** [Parte 01](./parte-01-introducao-ia-generativa.md) | ➡️ **Próximo:** [Parte 03 — APIs de LLMs](./parte-03-apis-de-llms.md)
