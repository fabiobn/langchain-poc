import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# Definição de ferramenta (tool)
@tool
def calcular_quadrado(numero: float) -> float:
    """Calcula o quadrado de um número."""
    return numero ** 2

# Definição do agente interno(worker)
tools_list = [calcular_quadrado]
llm_agente = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

agente_matematico_base = create_agent(llm_agente, tools=tools_list)

# Definição do estado do grafo
# Além das mensagens, controlamos variáveis de negócio (permissões e status)
class EstadoDoSistema(TypedDict):
    messages: Annotated[list, add_messages]
    status_atendimento: str  # Pode ser: "bot", "humano_pendente", "encerrado"
    alerta_seguranca: bool  # Flag de auditoria


# Definição dos nós do grafo(supervisor)
def node_auditoria(state: EstadoDoSistema):
    """Nó 1: Filtro de segurança em Python puro."""
    print("🛡️ [NÓ AUDITORIA] Analisando o input do usuário...")
    ultima_msg = state["messages"][-1].content.lower()

    # Exemplo simples de auditoria: bloquear palavras ofensivas fictícias
    palavras_bloqueadas = ["hackear", "burlar", "ofensa"]
    alerta = any(palavra in ultima_msg for palavra in palavras_bloqueadas)

    if alerta:
        print("⚠️ [AUDITORIA] Palavra bloqueada detectada!")
        return {
            "alerta_seguranca": True,
            "messages": [AIMessage(content="Seu input violou as políticas de segurança. Transferindo para supervisão.")]
        }

    return {"alerta_seguranca": False}


def node_agente_calculador(state: EstadoDoSistema):
    """Nó 2: Envelopa o agente gerado por create_agent."""
    print("🧠 [NÓ AGENTE MATEMÁTICO] Acionando agente especialista...")

    # Invocamos o agente base passando o histórico atual de mensagens do macro-grafo
    resultado_agente = agente_matematico_base.invoke({"messages": state["messages"]})

    # Capturamos a última resposta que o agente gerou
    ultima_resposta_bot = resultado_agente["messages"][-1].content

    # Se o bot não souber responder ou o usuário pedir um humano, mudamos o status
    status = "bot"
    if "humano" in state["messages"][-1].content.lower() or "falar com pessoa" in state["messages"][-1].content.lower():
        status = "humano_pendente"

    return {
        "messages": [AIMessage(content=ultima_resposta_bot)],
        "status_atendimento": status
    }


def node_suporte_humano(state: EstadoDoSistema):
    """Nó 3: Simula o transbordo para um atendente humano real."""
    print("🧑‍💻 [NÓ SUPORTE HUMANO] Aguardando intervenção de um atendente...")
    return {
        "messages": [AIMessage(
            content="[Suporte Humano]: Olá! Sou um supervisor humano. Como posso te ajudar com o problema que o robô não resolveu?")]
    }


# Definição de roteador para conditional edge
def roteador_inicial(state: EstadoDoSistema) -> Literal["ir_para_humano", "ir_para_agente"]:
    """Decide se manda o usuário para o bot ou direto para o suporte humano."""
    if state.get("alerta_seguranca") == True or state.get("status_atendimento") == "humano_pendente":
        return "ir_para_humano"
    return "ir_para_agente"


# Montagem do grafo maior(supervisor)
workflow = StateGraph(EstadoDoSistema)

# Adiciona todos os nós da nossa empresa no fluxo
workflow.add_node("auditoria", node_auditoria)
workflow.add_node("agente_calculador", node_agente_calculador)
workflow.add_node("suporte_humano", node_suporte_humano)

# O fluxo SEMPRE entra primeiro pela auditoria de segurança
workflow.add_edge(START, "auditoria")

# Saindo da auditoria, usamos o roteador condicional para decidir o rumo
workflow.add_conditional_edges(
    "auditoria",
    roteador_inicial,
    {
        "ir_para_humano": "suporte_humano",
        "ir_para_agente": "agente_calculador"
    }
)

# Se o fluxo for para o bot ou para o humano, eles encerram a rodada atual do grafo
# e esperam o próximo input do usuário
workflow.add_edge("agente_calculador", END)
workflow.add_edge("suporte_humano", END)

# URL Redis
REDIS_URI = os.getenv("REDIS_URL", "redis://localhost:6379")

configuracao_ttl = {
    "default_ttl": 60,      # Tempo de expiração definido em MINUTOS (ex: 60 minutos / 1 hora)
    "refresh_on_read": True # Sliding Expiration: Toda vez que o usuário mandar uma nova mensagem
                            # ou o script ler o histórico, o contador reseta para +60 minutos
}

with RedisSaver.from_conn_string(REDIS_URI, ttl=configuracao_ttl) as checkpointer:
    checkpointer.setup()

    # Compilamos o macro-grafo amarrando o Redis para salvar o estado de todos os nós
    sistema_corporativo = workflow.compile(checkpointer=checkpointer)

    print("\n==================================================")
    print("🏢 SISTEMA CORPORATIVO DE AGENTES INICIADO (REDIS)")
    print("==================================================")
    session_id = input("Número do Atendimento (Thread ID): ").strip().lower() or "atendimento_1"
    config = {"configurable": {"thread_id": session_id}}

    while True:
        try:
            user_input = input(f"\n[{session_id}] Cliente: ")
            if user_input.lower().strip() == 'sair':
                break
            if not user_input.strip():
                continue

            # Executa o macro-grafo
            estado_final = sistema_corporativo.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Imprime na tela o texto final produzido pelo nó vencedor do fluxo
            print(f"Sistema: {estado_final['messages'][-1].content}")

        except KeyboardInterrupt:
            break
