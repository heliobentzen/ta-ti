# Prática 01 — Primeiros Passos com APIs de LLMs

> **Carga horária estimada:** 2 horas  
> **Conteúdo relacionado:** [Parte 01](../conteudo/parte-01-introducao-ia-generativa.md), [Parte 02](../conteudo/parte-02-llms-como-funcionam.md), [Parte 03](../conteudo/parte-03-apis-de-llms.md)

---

## 🎯 Objetivos

Ao final desta prática, você será capaz de:
- Configurar o ambiente de desenvolvimento com as bibliotecas de IA
- Fazer sua primeira chamada à API de um LLM
- Entender o formato de mensagens (chat format)
- Implementar um chatbot de linha de comando simples com histórico

---

## 🔧 Configuração do Ambiente

### 1. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Instalar dependências

```bash
pip install openai python-dotenv
```

### 3. Configurar chave de API

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sk-...sua-chave-aqui...
```

> ⚠️ **NUNCA** commite o arquivo `.env` no repositório! Adicione ao `.gitignore`.

---

## 📝 Exercício 1 — Primeira Chamada à API

Crie o arquivo `pratica01/ex01_primeira_chamada.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Criar cliente
client = OpenAI()

# Fazer uma chamada simples
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explique o que é Python em exatamente 2 frases."
        }
    ]
)

# Extrair e exibir a resposta
print("=== Resposta do modelo ===")
print(response.choices[0].message.content)
print()
print("=== Metadados ===")
print(f"Modelo usado: {response.model}")
print(f"Tokens de entrada: {response.usage.prompt_tokens}")
print(f"Tokens de saída: {response.usage.completion_tokens}")
print(f"Total de tokens: {response.usage.total_tokens}")
```

**Execute e observe:**
```bash
python pratica01/ex01_primeira_chamada.py
```

**Reflexão:** O que acontece se você executar o código duas vezes? As respostas são iguais?

---

## 📝 Exercício 2 — Explorando Parâmetros

Crie `pratica01/ex02_parametros.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

prompt = "Escreva um haiku sobre programação."

print("=== Temperatura 0.0 (determinístico) ===")
for i in range(2):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    print(f"Chamada {i+1}: {resp.choices[0].message.content}")
    print()

print("=== Temperatura 1.5 (criativo) ===")
for i in range(2):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5
    )
    print(f"Chamada {i+1}: {resp.choices[0].message.content}")
    print()
```

**Questões para responder:**
1. As chamadas com `temperature=0.0` produzem respostas idênticas?
2. E com `temperature=1.5`?
3. O que isso implica para sistemas que precisam de respostas consistentes?

---

## 📝 Exercício 3 — O Poder do System Prompt

Crie `pratica01/ex03_system_prompt.py`:

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

pergunta = "O que é machine learning?"

personas = [
    {
        "nome": "Professor Didático",
        "system": "Você é um professor universitário especializado em explicar conceitos complexos de forma simples. Use analogias do cotidiano."
    },
    {
        "nome": "Especialista Técnico", 
        "system": "Você é um cientista de dados sênior. Seja técnico e preciso, use terminologia correta."
    },
    {
        "nome": "Criança de 10 anos",
        "system": "Você é uma criança de 10 anos muito curiosa que aprendeu sobre tecnologia. Explique como se fosse para um amigo da sua idade."
    }
]

for persona in personas:
    print(f"\n{'='*50}")
    print(f"PERSONA: {persona['nome']}")
    print('='*50)
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": persona["system"]},
            {"role": "user", "content": pergunta}
        ]
    )
    
    print(resp.choices[0].message.content)
```

---

## 📝 Exercício 4 — Chatbot com Histórico (Projeto Principal)

Crie `pratica01/chatbot.py` — um chatbot de linha de comando completo:

```python
from openai import OpenAI
from dotenv import load_dotenv
import sys

load_dotenv()

class Chatbot:
    def __init__(self, system_prompt: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.messages = []
        self.total_tokens = 0
        
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def chat(self, user_message: str) -> str:
        """Envia uma mensagem e retorna a resposta."""
        self.messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        self.total_tokens += response.usage.total_tokens
        
        return assistant_message
    
    def limpar(self):
        """Limpa o histórico, mantendo apenas o system prompt."""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msgs
        print("🗑️  Histórico limpo!")
    
    def historico(self):
        """Exibe o histórico da conversa."""
        print("\n=== HISTÓRICO ===")
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"[{role}]: {content}")
        print(f"Total de tokens usados: {self.total_tokens}")
        print("=================\n")


def main():
    print("🤖 Chatbot IA - IFPE Tópicos Avançados em TI")
    print("Comandos: '/limpar' limpa histórico | '/historico' mostra histórico | '/sair' encerra")
    print()
    
    system_prompt = input("System prompt (Enter para padrão): ").strip()
    if not system_prompt:
        system_prompt = "Você é um assistente útil e didático."
    
    bot = Chatbot(system_prompt=system_prompt)
    print(f"\n✅ Chatbot iniciado com model: gpt-4o-mini\n")
    
    while True:
        try:
            user_input = input("Você: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/sair":
                print(f"\nAté logo! Tokens totais usados: {bot.total_tokens}")
                break
            elif user_input == "/limpar":
                bot.limpar()
            elif user_input == "/historico":
                bot.historico()
            else:
                print("🤖 Assistente: ", end="", flush=True)
                response = bot.chat(user_input)
                print(response)
                print()
        
        except KeyboardInterrupt:
            print("\n\nEncerrando...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
```

**Execute:**
```bash
python pratica01/chatbot.py
```

---

## 🏆 Desafios Opcionais

1. **Streaming**: Modifique o chatbot para exibir a resposta token por token (use `stream=True`)
2. **Limite de contexto**: Implemente uma janela deslizante que mantém apenas as últimas N mensagens
3. **Múltiplos modelos**: Adicione um comando `/modelo` para trocar entre modelos durante a conversa
4. **Salvar conversa**: Adicione um comando `/salvar` que salva o histórico em um arquivo JSON

---

## ✅ Checklist de Entrega

- [ ] Exercício 1 funcionando (primeira chamada)
- [ ] Exercício 2 com comparação de temperaturas documentada
- [ ] Exercício 3 com 3 personas diferentes
- [ ] Chatbot com histórico funcional
- [ ] Pelo menos 1 desafio opcional implementado
- [ ] Arquivo `.env` no `.gitignore`

---

➡️ **Próxima prática:** [Prática 02 — Prompt Engineering](./pratica-02-prompt-engineering.md)
