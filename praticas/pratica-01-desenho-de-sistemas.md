# Prática 01 — Desenho de Sistemas com IA Generativa

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 01](../conteudo/parte-01-ia-generativa-desenho-sistemas.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Analisar requisitos e decidir quando (e se) usar IA Generativa
- Definir fronteiras do sistema entre componentes de IA e lógica tradicional
- Estimar custos de operação baseados em volume e modelo escolhido
- Documentar arquitetura de um sistema com componentes de IA

---

## 📝 Exercício 1 — Análise de Viabilidade: Usar ou Não Usar IA?

Para cada cenário abaixo, decida se IA Generativa é a abordagem correta. Justifique sua resposta com base nos critérios: **custo**, **latência**, **determinismo** e **valor agregado**.

**Cenários:**

1. Um sistema bancário que precisa calcular juros sobre um empréstimo.
2. Um suporte ao cliente que responde perguntas abertas sobre produtos de uma loja virtual.
3. Um sistema que valida se um CPF tem o formato correto.
4. Uma ferramenta que resume automaticamente relatórios internos de 50 páginas enviados por gerentes.
5. Um jogo que precisa gerar diálogos únicos para personagens NPCs.
6. Um sistema que verifica se um email está no banco de dados de usuários.

**Para cada cenário, preencha a tabela:**

| Cenário | Usar IA? | Alternativa (se não) | Justificativa |
|---------|----------|----------------------|---------------|
| 1 | | | |
| 2 | | | |
| ... | | | |

---

## 📝 Exercício 2 — Mapeando Fronteiras do Sistema

Escolha **um** dos cenários abaixo e desenhe (em texto, ASCII art ou diagrama Mermaid) a arquitetura do sistema, identificando claramente:
- Quais componentes usam IA Generativa
- Quais usam lógica tradicional (código, banco de dados, APIs)
- O fluxo de dados entre eles

**Cenário A — Assistente de Triagem Médica:**  
Um sistema web onde pacientes descrevem sintomas em linguagem natural e recebem orientações iniciais sobre urgência do atendimento.

**Cenário B — Gerador de Relatórios de Vendas:**  
Um sistema que lê dados de um banco de dados e gera narrativas em linguagem natural sobre o desempenho mensal da equipe.

**Cenário C — Revisor de Código Automático:**  
Uma ferramenta de CI/CD que analisa pull requests e faz comentários sobre qualidade, segurança e boas práticas.

**Exemplo de diagrama Mermaid:**
```
graph TD
    A[Usuário] -->|mensagem de texto| B[Frontend]
    B -->|POST /chat| C[API Backend]
    C -->|valida sessão| D[(Banco de Dados)]
    C -->|envia prompt| E[LLM API]
    E -->|resposta gerada| C
    C -->|resposta| B
    B -->|exibe| A
```

---

## 📝 Exercício 3 — Estimativa de Custos

Você vai construir o **Assistente de Triagem Médica** (Cenário A do Exercício 2).

**Dados do sistema:**
- Volume esperado: 500 consultas por dia
- Tamanho médio da mensagem do usuário: 150 tokens
- Tamanho médio do system prompt: 500 tokens
- Tamanho médio da resposta do sistema: 300 tokens
- Histórico médio mantido por sessão: 3 turnos anteriores (antes da mensagem atual)

**Tabela de preços (valores ilustrativos para o exercício):**

| Modelo | Input (por 1M tokens) | Output (por 1M tokens) |
|--------|----------------------|------------------------|
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-4o | $2.50 | $10.00 |
| Claude 3 Haiku | $0.25 | $1.25 |
| Llama 3.2 (local) | $0.00 | $0.00 |

**Tarefas:**

1. Calcule o total de tokens de entrada por consulta (system prompt + histórico + mensagem atual).
2. Calcule o custo diário e mensal (30 dias) para cada modelo.
3. Qual modelo você escolheria? Por quê? Considere custo, qualidade e privacidade dos dados médicos.
4. A que volume de consultas diárias o modelo local se tornaria obrigatório por questão de custo (assuma infraestrutura de servidor com custo de $200/mês)?

---

## 📝 Exercício 4 — Documento de Arquitetura (Projeto Principal)

Crie um documento `pratica01/arquitetura.md` descrevendo o sistema que você escolheu no Exercício 2. O documento deve conter:

```markdown
# Arquitetura: [Nome do Sistema]

## 1. Problema a Resolver
[Descrição em 3-5 frases do problema de negócio]

## 2. Decisão: Por que IA Generativa?
[Justificativa clara. Quais alternativas foram consideradas e por que foram descartadas?]

## 3. Componentes do Sistema

### Componentes com IA Generativa
- **[Nome]:** [Responsabilidade e modelo escolhido]

### Componentes Tradicionais
- **[Nome]:** [Tecnologia e responsabilidade]

## 4. Fluxo de Dados
[Diagrama em texto ou Mermaid]

## 5. Estimativa de Custo
[Cálculo baseado no Exercício 3]

## 6. Riscos e Mitigações
| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Resposta incorreta do LLM | Alta | Alto | Revisão humana obrigatória |
| ... | | | |

## 7. Limitações Conhecidas
[O que o sistema NÃO fará e por quê]
```

---

## 🏆 Desafios Opcionais

1. **Comparação com abordagem tradicional:** Para o Cenário B (relatórios de vendas), estime quanto tempo um analista levaria para escrever o relatório manualmente vs. o custo da solução com IA. A IA se paga em quanto tempo?

2. **Arquitetura com fallback:** Redesenhe seu sistema adicionando um mecanismo de fallback — o que acontece se a API do LLM estiver indisponível? Como o sistema deve se comportar?

3. **Considerações de privacidade:** Pesquise a política de dados da OpenAI e da Anthropic. Para o cenário médico, quais dados não podem ser enviados para APIs externas? Como você redesenharia o sistema para ser LGPD-compliant?

---

## ✅ Checklist de Entrega

- [ ] Exercício 1: tabela de viabilidade preenchida com justificativas
- [ ] Exercício 2: diagrama de arquitetura com fronteiras claramente marcadas
- [ ] Exercício 3: cálculo de custos para todos os modelos
- [ ] Exercício 4: documento `arquitetura.md` completo
- [ ] Pelo menos 1 desafio opcional

---

➡️ **Próxima prática:** [Prática 02 — LLMs e Contexto](./pratica-02-llms-e-contexto.md)
