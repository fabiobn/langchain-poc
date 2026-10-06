from dotenv import load_dotenv
from langchain.agents import create_agent  # Importação moderna
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()


@tool
def calcular_idade_pet(anos_humanos: int) -> int:
  """Calcula a idade de um cachorro em anos de cachorro (multiplica por 7)."""
  return anos_humanos * 7


ferramentas = [calcular_idade_pet]
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Estrutura do novo prompt (não precisa mais do 'agent_scratchpad')
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil e preciso."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

# Criando o agente moderno
agente = create_agent(
    model=llm,
    tools=ferramentas,
    system_prompt="Você é um assistente útil e preciso." # <-- Correção aqui
)

# 5. Executar o agente enviando a mensagem no formato correto
resposta = agente.invoke({
    "messages": [
        {"role": "user", "content": "Quantos anos tem um cachorro de 4 anos em idade de cão?"}
    ]
})

# O LangChain moderno retorna uma lista de mensagens; a última será a resposta do agente
print("\nResposta Final:", resposta["messages"][-1].content)
