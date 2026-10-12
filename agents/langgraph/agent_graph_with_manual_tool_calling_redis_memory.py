import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver

load_dotenv()

# Definição da Ferramenta (Tool)
@tool
def calcular_quadrado(numero: float) -> float:
    """Calcula o quadrado de um número. Use esta ferramenta sempre que
    o usuário pedir o quadrado ou potência de 2.
    """
    return numero ** 2


tools_list = [calcular_quadrado]
# Registro manual para podermos invocar a função Python pelo nome que a IA pedir
tools_registry = {"calcular_quadrado": calcular_quadrado}


# Definição do Estado do Grafo
# O Estado dita a estrutura de dados que trafega entre os nós.
# 'add_messages' é o reducer que anexa novas mensagens ao histórico de forma segura.
class AgenteEstado(TypedDict):
    messages: Annotated[list, add_messages]


# Inicialização do Modelo acoplado com as Ferramentas
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm_with_tools = llm.bind_tools(tools_list)


def node_ia(state: AgenteEstado) -> AgenteEstado:
    """Nó da IA: Recebe o estado atual de mensagens, pensa e toma uma decisão."""
    print("\n🧠 [NÓ IA] Avaliando contexto e decidindo próximo passo...")
    resposta = llm_with_tools.invoke(state["messages"])
    # Retorna um dicionário com a chave 'messages' para o reducer anexar ao estado
    return {"messages": [resposta]}


def node_ferramentas(state: AgenteEstado) -> AgenteEstado:
    """Nó de Ferramentas: Executa a função Python real se o nó da IA tiver solicitado."""
    print("⚙️ [NÓ FERRAMENTAS] Executando comando Python local...")
    ultima_mensagem = state["messages"][-1]

    lista_de_retornos = []
    # Itera sobre os pedidos de ferramentas feitos pelo LLM
    for tool_call in ultima_mensagem.tool_calls:
        func_nome = tool_call["name"]
        func_args = tool_call["args"]
        tool_id = tool_call["id"]

        # Executa a função do nosso registro
        funcao_alvo = tools_registry[func_nome]
        resultado_python = funcao_alvo.invoke(func_args)

        # Cria a mensagem ToolMessage para alimentar o grafo de volta
        lista_de_retornos.append(
            ToolMessage(content=str(resultado_python), tool_call_id=tool_id)
        )

    return {"messages": lista_de_retornos}


def roteador_condicional(state: AgenteEstado) -> Literal["ir_para_ferramentas", "ir_para_fim"]:
    """Examina a última resposta do LLM para rotear o fluxo."""
    ultima_mensagem = state["messages"][-1]

    # Se a IA devolveu intenções de chamadas de ferramenta (tool_calls), mudamos o rumo do fluxo
    if hasattr(ultima_mensagem, "tool_calls") and ultima_mensagem.tool_calls:
        return "ir_para_ferramentas"

    # Caso contrário, a conversa com o usuário encerrou nesta iteração
    return "ir_para_fim"


# Inicializa o construtor de grafo de estados
workflow = StateGraph(AgenteEstado)

# Adiciona os nós criados ao grafo
workflow.add_node("ia_node", node_ia)
workflow.add_node("tools_node", node_ferramentas)

# Mapeia as arestas fixas e direcionais
workflow.add_edge(START, "ia_node")  # O ponto de entrada sempre vai direto para o nó IA
workflow.add_edge("tools_node", "ia_node")  # Após rodar a ferramenta, o fluxo SEMPRE volta para o nó IA

# Configura o roteamento condicional saindo do nó IA
workflow.add_conditional_edges(
    "ia_node",
    roteador_condicional,
    {
        "ir_para_ferramentas": "tools_node",  # Se retornar a string correspondente, vai rodar a ferramenta
        "ir_para_fim": END  # Se não precisar de ferramentas, envia para o nó final
    }
)

# URL Redis
REDIS_URI = os.getenv("REDIS_URL", "redis://localhost:6379")

configuracao_ttl = {
    "default_ttl": 60,      # Tempo de expiração definido em MINUTOS (ex: 60 minutos / 1 hora)
    "refresh_on_read": True # Sliding Expiration: Toda vez que o usuário mandar uma nova mensagem
                            # ou o script ler o histórico, o contador reseta para +60 minutos
}

print("==================================================")
print("🤖 GRAFO CUSTOMIZADO (STATEGRAPH) OPERANDO EM REDIS")
print("==================================================")

with RedisSaver.from_conn_string(REDIS_URI, ttl=configuracao_ttl) as checkpointer:
    # Garante os índices do motor Redis Search ativo
    checkpointer.setup()

    # Compila o grafo acoplando o persistidor Redis de forma nativa
    agente_grafo = workflow.compile(checkpointer=checkpointer)

    session_id = input("Identificador de Usuário (Thread ID): ").strip().lower() or "default_user"
    config = {"configurable": {"thread_id": session_id}}

    print(f"\nGrafo customizado compilado e monitorando a thread: '{session_id}'.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        try:
            user_input = input(f"[{session_id}] Você: ")
            if user_input.lower().strip() == 'sair':
                break
            if not user_input.strip():
                continue

            # Dispara a execução do grafo enviando a nova entrada do usuário
            # O RedisSaver vai interceptar, salvar os checkpoints de cada nó e recuperar históricos automaticamente
            estado_final = agente_grafo.invoke(
                {"messages": [("user", user_input)]},
                config=config
            )

            # Recupera o conteúdo do último nó processado no estado final do grafo
            resposta_final = estado_final["messages"][-1].content
            print(f"IA: {resposta_final}")

        except KeyboardInterrupt:
            break
