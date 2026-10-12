import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
import redis
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import AIMessage, HumanMessage

# Carrega as variáveis do arquivo .env
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Por favor, configure a GROQ_API_KEY no arquivo .env")


# Ferramenta (tool) do agente interno (worker)
@tool
def calcular_quadrado(numero: float) -> float:
    """Calcula o quadrado de um número."""
    return numero ** 2


tools_list = [calcular_quadrado]

# Agente interno (worker)
llm_agente = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
agente_matematico_base = create_agent(llm_agente, tools=tools_list)


# Definirção do estado do grafo (supervisor)
class EstadoDoSistema(TypedDict):
    messages: Annotated[list, add_messages]
    status_atendimento: Literal["bot", "humano_pendente", "encerrado"]
    alerta_seguranca: bool
    resposta_operador: str


# Definição dos nós do grafo (supervisor)
def node_auditoria(state: EstadoDoSistema):
    """Nó 1: Analisa o input do usuário e gerencia as regras de transbordo determinísticas."""
    print("🛡️ [NÓ AUDITORIA] Analisando o input do usuário...")
    ultima_msg = state["messages"][-1].content.lower()

    # Regra 1: Filtro de segurança (Palavras bloqueadas)
    palavras_bloqueadas = ["hackear", "burlar", "ofensa"]
    if any(palavra in ultima_msg for palavra in palavras_bloqueadas):
        print("⚠️ [AUDITORIA] Alerta de segurança acionado!")
        return {
            "alerta_seguranca": True,
            "status_atendimento": "humano_pendente",
            "messages": [AIMessage(content="[Sistema]: Seu input disparou um gatilho de segurança.")]
        }

    # Regra 2: INTENÇÃO DE HUMANO RESTRITA (Garante o transbordo imediato sem passar pelo LLM)
    palavras_transbordo = ["pessoa", "humano", "atendente", "central", "telefone"]
    if any(palavra in ultima_msg for palavra in palavras_transbordo):
        print("🧑‍💻 [AUDITORIA] Intenção de suporte humano detectada. Redirecionando...")
        return {
            "alerta_seguranca": False,
            "status_atendimento": "humano_pendente"
        }

    return {"alerta_seguranca": False, "status_atendimento": "bot"}


def node_agente_calculador(state: EstadoDoSistema):
    """Nó 2: Executa o bot apenas se o usuário não pediu transbordo."""
    print("🧠 [NÓ AGENTE MATEMÁTICO] Acionando especialista Groq...")

    resultado_agente = agente_matematico_base.invoke({"messages": state["messages"]})
    ultima_resposta_bot = resultado_agente["messages"][-1].content

    return {
        "messages": [AIMessage(content=ultima_resposta_bot)],
        "status_atendimento": "bot"
    }


def node_suporte_humano(state: EstadoDoSistema):
    """Nó 3: Roda quando o operador humano insere o texto e o grafo é retomado."""
    print("🧑‍💻 [NÓ SUPORTE HUMANO] Injetando a resposta do atendente no Redis...")

    # Recupera o texto capturado durante a pausa
    texto_humano = state.get("resposta_operador", "Suporte humano indisponível.")

    # AIMessage explícita para o LangGraph renderizar ao cliente
    return {
        "messages": [AIMessage(content=f"[Atendente Humano]: {texto_humano}")],
        "status_atendimento": "bot",
        "resposta_operador": ""  # Limpa o buffer para a próxima mensagem
    }


# Função de roteamento do conditional edge
def roteador_inicial(state: EstadoDoSistema) -> Literal["ir_para_humano", "ir_para_agente"]:
    # Se a auditoria marcou o status como humano_pendente, pula o bot e vai pro nó humano
    if state.get("status_atendimento") == "humano_pendente":
        return "ir_para_humano"
    return "ir_para_agente"


# Definição do grafo
workflow = StateGraph(EstadoDoSistema)

workflow.add_node("auditoria", node_auditoria)
workflow.add_node("agente_calculador", node_agente_calculador)
workflow.add_node("suporte_humano", node_suporte_humano)

workflow.add_edge(START, "auditoria")

workflow.add_conditional_edges(
    "auditoria",
    roteador_inicial,
    {
        "ir_para_humano": "suporte_humano",
        "ir_para_agente": "agente_calculador"
    }
)

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

    # Congela antes de entrar na execução do nó humano
    sistema_corporativo = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["suporte_humano"]
    )

    print("\n==================================================")
    print("🏢 SISTEMA CORPORATIVO FIEL AO CONTEXTO SUPERVISOR-WORKER E HUMAN-IN-THE-LOOP")
    print("==================================================")
    session_id = input("Número do Atendimento (Thread ID): ").strip().lower() or "atendimento_1"
    config = {"configurable": {"thread_id": session_id}}

    redis.Redis.from_url(REDIS_URI).delete(f"chat_history:{session_id}")

    while True:
        try:
            user_input = input(f"\n[{session_id}] Cliente: ")
            if user_input.lower().strip() == 'sair':
                break
            if not user_input.strip():
                continue

            # Executa o grafo supervisor
            estado_atual = sistema_corporativo.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Checa se o Redis pausou antes do nó humano
            estado_do_grafo = sistema_corporativo.get_state(config)

            if estado_do_grafo.next and "suporte_humano" in estado_do_grafo.next:
                print("\n🛑 [PAUSA DO GRAFO] Encaminhado para a fila de suporte humano!")

                resposta_atendente = input("🧑‍💻 [PAINEL DO ATENDENTE] Responda ao cliente: ")

                # Injeta a resposta do atendente no estado
                sistema_corporativo.update_state(
                    config,
                    {"resposta_operador": resposta_atendente},
                    as_node="suporte_humano"
                )

                print("✅ Retomando do checkpoint do Redis...")
                # Retoma a execução
                estado_atual = sistema_corporativo.invoke(None, config=config)

            # Última mensagem do estado atualizado
            print(f"Sistema: {estado_atual['messages'][-1].content}")

        except KeyboardInterrupt:
            break