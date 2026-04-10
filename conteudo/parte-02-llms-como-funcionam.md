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

## 2.9 Experimentando com LLMs Localmente (Gratuito)

Para entender melhor como LLMs funcionam na prática, é essencial experimentar. Com **Ollama**, você pode rodar modelos open-source localmente, sem custo e sem necessidade de API keys.

### Instalação do Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Baixar um modelo leve para experimentação
ollama pull llama3.2       # 3B parâmetros, ~2GB
ollama pull phi3:mini       # 3.8B parâmetros, ~2.3GB
```

### Explorando a Geração de Tokens

```python
# Demonstração: ver o LLM gerando token por token
from openai import OpenAI

# Conectar ao Ollama (API compatível com OpenAI)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Geração com streaming — observe a geração autorregressiva token a token
stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explique o que é atenção em Transformers."}],
    stream=True,
    temperature=0.7
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Experimentando com Temperatura

```python
# Observe como a temperatura afeta a geração
for temp in [0.0, 0.5, 1.0, 1.5]:
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Complete: O gato sentou no..."}],
        temperature=temp,
        max_tokens=20
    )
    print(f"Temp {temp}: {response.choices[0].message.content}")
```

> **💡 Dica educacional:** Rodar modelos localmente permite experimentar livremente sem custo. Use Ollama durante todo o curso como alternativa gratuita às APIs comerciais.

---

## 2.10 Construindo a Intuição — "LLM do Zero"

Para realmente entender como um LLM funciona, é valioso ver a construção passo a passo. Andrej Karpathy (ex-diretor de IA da Tesla) disponibiliza gratuitamente uma série de vídeos onde constrói um modelo de linguagem do zero.

### O Processo Simplificado

```
1. DADOS          → Coletar textos (ex: obras de Shakespeare)
2. TOKENIZAÇÃO    → Converter texto em números (vocabulário)
3. EMBEDDING      → Números → vetores densos (aprendem significado)
4. TRANSFORMER    → Camadas de atenção + feed-forward
5. TREINAMENTO    → Prever próximo token, ajustar pesos via backpropagation
6. GERAÇÃO        → Dado um contexto, amostrar próximo token repetidamente
```

### Analogia Intuitiva

Imagine que você leu milhões de livros e alguém começa uma frase:

> "O cientista entrou no laboratório e viu que o experimento..."

Seu cérebro automaticamente calcula as continuações prováveis:
- "...tinha dado certo" (35%)
- "...estava em andamento" (25%)
- "...falhou novamente" (20%)
- "...desapareceu misteriosamente" (10%)
- ...outras opções (10%)

Um LLM faz **exatamente isso**, mas de forma matemática: calcula uma distribuição de probabilidade sobre todo o vocabulário e amostra o próximo token. A temperatura controla se ele escolhe sempre a opção mais provável (conservador) ou explora opções menos comuns (criativo).

### Modelos Open-Source para Estudo

| Modelo | Parâmetros | Onde Obter | Destaque |
|--------|-----------|------------|----------|
| Llama 3.2 | 1B / 3B | Ollama, Hugging Face | Leve, roda em qualquer máquina |
| Phi-3 Mini | 3.8B | Ollama, Hugging Face | Alta qualidade para seu tamanho |
| Mistral 7B | 7B | Ollama, Hugging Face | Excelente relação qualidade/tamanho |
| DeepSeek R1 (distill) | 1.5B / 7B | Ollama, Hugging Face | Raciocínio avançado, open-source |
| Gemma 2 | 2B / 9B | Ollama, Hugging Face | Google, bom para experimentação |

> **🎓 Para aprofundar:** Assista [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) de Andrej Karpathy (gratuito no YouTube). É a melhor forma de construir intuição sobre como Transformers e LLMs realmente funcionam.

---

## 2.11 Limitações Técnicas

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
| Ollama | Ferramenta para rodar LLMs localmente, sem custo |
| Modelos open-source | Llama, Phi, Mistral, DeepSeek — gratuitos para estudo |

---

## 🔗 Referências

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Training language models to follow instructions with human feedback (RLHF)](https://arxiv.org/abs/2203.02155)
- [Let's build GPT: from scratch — Andrej Karpathy (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Ollama — Rodar LLMs localmente](https://ollama.ai)
- [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

⬅️ **Anterior:** [Parte 01](./parte-01-introducao-ia-generativa.md) | ➡️ **Próximo:** [Parte 03 — APIs de LLMs](./parte-03-apis-de-llms.md)
