import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Definição da Ferramenta (Tool)
@tool
def calcular_quadrado(numero: float) -> float:
    """Calcula o quadrado de um número. Use esta ferramenta sempre que
    o usuário pedir o quadrado, potência de 2 ou multiplicação de um número por si mesmo.
    """
    return numero ** 2

# Agrupamos as ferramentas que o agente terá acesso
tools = [calcular_quadrado]

# Inicialização do Modelo de Linguagem
# Usamos temperatura 0 para garantir respostas determinísticas nas chamadas de ferramentas
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Criação do Agente
# Esse wrapper envelopa o modelo e as ferramentas, gerenciando o ciclo de execução automaticamente
agente = create_agent(llm, tools=tools)

# Execução de Teste
print("🤖 Enviando comando para o agente...")

pergunta = "Quanto é o quadrado de 12?"
inputs = {"messages": [("user", pergunta)]}

# O agente retorna o fluxo completo de mensagens que trafegaram pelo grafo
resultado = agente.invoke(inputs)

# Exibimos apenas a última mensagem do fluxo, que contém a resposta interpretada da IA
print(f"Usuário: {pergunta}")
print(f"Agente: {resultado['messages'][-1].content}")
