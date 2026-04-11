# Prática 03 — Engenharia de Contexto I: Instruções, Papéis e Saída Estruturada

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 03](../conteudo/parte-03-engenharia-de-contexto-1.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Aplicar técnicas de prompt engineering em cenários reais
- Comparar resultados de diferentes abordagens de prompt
- Construir um sistema de extração de dados estruturados
- Implementar chain-of-thought para problemas complexos

---

## 📝 Exercício 1 — Zero-Shot vs Few-Shot

Crie `pratica02/ex01_zero_few_shot.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

textos_teste = [
    "Adorei o produto! Chegou antes do prazo e a qualidade é incrível.",
    "Péssima experiência. Produto com defeito e atendimento horrível.",
    "Produto ok. Não é o que esperava mas serve para o que precisava.",
    "Excelente! Superou minhas expectativas em tudo.",
    "Entrega demorou 3 semanas além do prazo. Não recomendo.",
]

def classificar_zero_shot(texto: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classifique o sentimento deste review como 
            POSITIVO, NEGATIVO ou NEUTRO.
            Retorne apenas a classificação, sem mais texto.
            
            Review: {texto}"""
        }],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def classificar_few_shot(texto: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classifique o sentimento como POSITIVO, NEGATIVO ou NEUTRO.
            Retorne apenas a classificação.
            
            Review: "Produto chegou rápido e funciona perfeitamente!"
            Sentimento: POSITIVO
            
            Review: "Não funcionou desde o primeiro dia. Pedi reembolso."
            Sentimento: NEGATIVO
            
            Review: "É razoável pelo preço que paguei."
            Sentimento: NEUTRO
            
            Review: "Atendimento demorou mas resolveu o problema."
            Sentimento: NEUTRO
            
            Review: {texto}
            Sentimento:"""
        }],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Comparar
print(f"{'Texto':<55} {'Zero-Shot':<12} {'Few-Shot'}")
print("-" * 80)
for texto in textos_teste:
    zs = classificar_zero_shot(texto)
    fs = classificar_few_shot(texto)
    match = "✅" if zs == fs else "❌"
    print(f"{texto[:52]:<55} {zs:<12} {fs} {match}")
```

---

## 📝 Exercício 2 — Chain of Thought

Crie `pratica02/ex02_chain_of_thought.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

problemas = [
    "Se eu tenho 15 maçãs e dou 1/3 para minha irmã e metade do restante para meu irmão, quantas ficam comigo?",
    "Um trem parte às 8h30 e chega às 14h15. Faz uma parada de 45 minutos no caminho. Quanto tempo dura o trajeto sem parada?",
    "Uma loja tem desconto de 25% num produto de R$240. Com esse desconto compro 3 unidades. Quanto gasto no total?",
]

def resolver_simples(problema: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Resolva: {problema}"}],
        temperature=0
    )
    return resp.choices[0].message.content

def resolver_com_cot(problema: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Resolva o problema mostrando CADA PASSO do raciocínio.
            Seja explícito em todas as operações.
            Termine com "RESPOSTA FINAL: [resultado]"
            
            Problema: {problema}"""
        }],
        temperature=0
    )
    return resp.choices[0].message.content

for i, problema in enumerate(problemas, 1):
    print(f"\n{'='*60}")
    print(f"PROBLEMA {i}: {problema}")
    print(f"\n--- SEM CHAIN-OF-THOUGHT ---")
    print(resolver_simples(problema))
    print(f"\n--- COM CHAIN-OF-THOUGHT ---")
    print(resolver_com_cot(problema))
```

**Análise:** Em qual tipo de problema o Chain-of-Thought faz mais diferença?

---

## 📝 Exercício 3 — Extração de Dados Estruturados

Crie `pratica02/ex03_extracao_estruturada.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

textos = [
    """
    Entre em contato com João Silva pelo email joao.silva@empresa.com.br
    ou ligue para (81) 99876-5432. Ele é Gerente de TI na TechCorp Recife.
    """,
    """
    Prezada Maria Santos, conforme combinado, segue proposta para seu cargo
    de Analista de Dados. Você pode retornar pelo telefone 11-3456-7890
    ou pelo email maria@startup.io. Abraços, DataCo SP.
    """,
    """
    Nosso CEO Carlos Mendes (carlos@globaltech.com | +55 71 98765-4321)
    estará disponível na próxima semana para a reunião.
    """
]

def extrair_contato(texto: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Extraia as informações de contato do texto abaixo.
            Retorne SOMENTE um JSON válido com esta estrutura:
            {{
                "nome": "nome completo ou null",
                "email": "email ou null",
                "telefone": "telefone formatado ou null",
                "cargo": "cargo/título ou null",
                "empresa": "nome da empresa ou null",
                "cidade": "cidade ou null"
            }}
            
            Texto: {texto}"""
        }],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

for i, texto in enumerate(textos, 1):
    print(f"\n=== Texto {i} ===")
    resultado = extrair_contato(texto)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
```

---

## 📝 Exercício 4 — Sistema de Classificação de Suporte (Projeto Principal)

Crie `pratica02/sistema_suporte.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """Você é um sistema de triagem de tickets de suporte técnico.

Para cada ticket, retorne SOMENTE um JSON com:
{
  "categoria": "bug" | "feature" | "duvida" | "cobranca" | "outro",
  "prioridade": "critica" | "alta" | "media" | "baixa",
  "sentimento": "muito_insatisfeito" | "insatisfeito" | "neutro" | "satisfeito",
  "resumo": "resumo em máximo 15 palavras",
  "sugestao_resposta": "primeira linha de resposta sugerida",
  "palavras_chave": ["palavra1", "palavra2", "palavra3"],
  "precisa_escalada": true | false
}

Regras:
- "critica" = sistema fora do ar ou perda de dados
- "alta" = funcionalidade principal não funciona
- "media" = funcionalidade secundária com problema
- "baixa" = dúvida ou sugestão
- precisa_escalada = true se prioridade critica ou alta E sentimento muito_insatisfeito"""

tickets = [
    {
        "id": "TK001",
        "usuario": "empresa_xyz",
        "mensagem": "URGENTE! Nosso sistema está completamente fora do ar desde as 9h. Perdemos vendas de R$50.000. Isso é inaceitável!!!"
    },
    {
        "id": "TK002", 
        "usuario": "joao_dev",
        "mensagem": "Oi, gostaria de saber como exportar relatórios para PDF. Não encontrei essa opção no menu."
    },
    {
        "id": "TK003",
        "usuario": "maria_gestora",
        "mensagem": "O botão de aprovação não está funcionando no módulo financeiro. Consigo ver as solicitações mas não aprovar."
    },
    {
        "id": "TK004",
        "usuario": "carlos_admin",
        "mensagem": "Seria possível adicionar um filtro por data de criação na listagem de usuários? Facilitaria muito nosso trabalho."
    }
]

def processar_ticket(ticket: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket["mensagem"]}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    analise = json.loads(response.choices[0].message.content)
    return {**ticket, "analise": analise, "processado_em": datetime.now().isoformat()}

print("🎫 SISTEMA DE TRIAGEM DE TICKETS\n")
print("="*60)

resultados = []
for ticket in tickets:
    resultado = processar_ticket(ticket)
    resultados.append(resultado)
    
    a = resultado["analise"]
    emoji_prioridade = {"critica": "🔴", "alta": "🟠", "media": "🟡", "baixa": "🟢"}
    
    print(f"\n[{resultado['id']}] {emoji_prioridade.get(a['prioridade'], '⚪')} {a['prioridade'].upper()}")
    print(f"  Categoria: {a['categoria']}")
    print(f"  Sentimento: {a['sentimento']}")
    print(f"  Resumo: {a['resumo']}")
    print(f"  Escalar: {'⚠️ SIM' if a['precisa_escalada'] else 'Não'}")
    print(f"  Sugestão: {a['sugestao_resposta']}")

# Estatísticas
print("\n" + "="*60)
print("📊 ESTATÍSTICAS")
prioridades = [r["analise"]["prioridade"] for r in resultados]
for p in ["critica", "alta", "media", "baixa"]:
    count = prioridades.count(p)
    bar = "█" * count
    print(f"  {p:<8}: {bar} ({count})")
```

---

## 🏆 Desafios Opcionais

1. **A/B Testing de Prompts**: Implemente uma função que testa duas versões de prompt e compara a qualidade
2. **Prompt com Exemplos Dinâmicos**: Crie um sistema de few-shot onde os exemplos são selecionados com base na similaridade com a entrada
3. **Auto-avaliação**: Peça ao modelo para avaliar sua própria resposta e melhorá-la
4. **Multi-step**: Implemente um pipeline de 3 etapas: extração → validação → formatação

---

## ✅ Checklist de Entrega

- [ ] Ex01: zero-shot vs few-shot com análise das diferenças
- [ ] Ex02: chain-of-thought funcionando para os 3 problemas
- [ ] Ex03: extração estruturada gerando JSON válido
- [ ] Sistema de suporte classificando todos os tickets
- [ ] Pelo menos 1 desafio opcional

---

⬅️ **Anterior:** [Prática 02](./pratica-02-llms-e-contexto.md) | ➡️ **Próxima:** [Prática 04 — Engenharia de Contexto II](./pratica-04-engenharia-de-contexto-2.md)
