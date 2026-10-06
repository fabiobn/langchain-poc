from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Definição da Ferramenta (Tool)
@tool
def calcular_quadrado(numero: float) -> float:
    """Calcula o quadrado de um número. Use esta ferramenta sempre que
    o usuário pedir o quadrado ou potência de 2.
    """
    return numero ** 2


tools = [calcular_quadrado]

# Inicialização do LLM
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Ativação do Checkpointer
memory = MemorySaver()

# Criação do Agente
# Passamos o checkpointer nativo diretamente na criação
agente = create_agent(llm, tools=tools, checkpointer=memory)

print("==================================================")
print("🤖 AGENTE REACT COM MEMÓRIA POR THREADS")
print("==================================================")
print("Para testar o isolamento de múltiplos usuários, use as threads: 'carlos' ou 'ana'")

while True:
    try:
        # Thread atual
        thread_id = input("\n[Thread ID] Quem é você? (carlos / ana / sair): ").strip().lower()

        if thread_id == 'sair':
            print("🤖 Encerrando o sistema de agentes.")
            break

        if not thread_id:
            continue

        # Dicionário de configuração padrão que direciona a memória no LangGraph
        config = {"configurable": {"thread_id": thread_id}}

        # Captura a pergunta do usuário selecionado
        user_input = input(f"[{thread_id}] Digite seu comando: ")
        if not user_input.strip():
            continue

        # Envia a nova mensagem passando a configuração da thread correspondente
        inputs = {"messages": [("user", user_input)]}
        resultado = agente.invoke(inputs, config=config)

        # Exibe a resposta final gerada pelo nó de IA
        resposta_final = resultado["messages"][-1].content
        print(f"🤖 Agente para {thread_id}: {resposta_final}")

    except KeyboardInterrupt:
        print("\n🤖 Encerrado abruptamente.")
        break
