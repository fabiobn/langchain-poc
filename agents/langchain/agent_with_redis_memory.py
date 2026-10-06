import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.redis import RedisSaver

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

# URL Redis
REDIS_URI = os.getenv("REDIS_URL", "redis://localhost:6379")

configuracao_ttl = {
    "default_ttl": 60,      # Tempo de expiração definido em MINUTOS (ex: 60 minutos / 1 hora)
    "refresh_on_read": True # Sliding Expiration: Toda vez que o usuário mandar uma nova mensagem
                            # ou o script ler o histórico, o contador reseta para +60 minutos
}

print("==================================================")
print("🤖 AGENTE REACT COM PERSISTÊNCIA REAL EM REDIS")
print("==================================================")

# Gerenciador de Contexto do RedisSaver
# O 'with' garante que a conexão com o banco de dados seja aberta e fechada de forma limpa
with RedisSaver.from_conn_string(REDIS_URI, ttl=configuracao_ttl) as checkpointer:
    # Configura a estrutura interna do Redis na primeira inicialização
    # Cria os índices de busca e controle do LangGraph no banco
    checkpointer.setup()

    # Inicialização do Agente passando o checkpointer do Redis
    agente = create_agent(llm, tools=tools, checkpointer=checkpointer)

    print("Para testar, use as threads persistentes: 'carlos' ou 'ana'")

    while True:
        try:
            thread_id = input("\n[Thread ID] Quem é você? (carlos / ana / sair): ").strip().lower()
            if thread_id == 'sair':
                print("🤖 Desconectando do Redis e encerrando.")
                break

            if not thread_id:
                continue

            config = {"configurable": {"thread_id": thread_id}}

            user_input = input(f"[{thread_id}] Digite seu comando: ")
            if not user_input.strip():
                continue

            inputs = {"messages": [("user", user_input)]}
            resultado = agente.invoke(inputs, config=config)

            resposta_final = resultado["messages"][-1].content
            print(f"🤖 Agente para {thread_id}: {resposta_final}")

        except KeyboardInterrupt:
            print("\n🤖 Encerrado abruptamente.")
            break
