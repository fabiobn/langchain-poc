import os
import json
import requests
from dotenv import load_dotenv
import redis
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


load_dotenv()


# Definição das tools
# O LangChain lê as Docstrings e a Tipagem do Python para instruir o LLM
@tool
def get_current_weather(city: str) -> str:
    """Busca o clima em tempo real de uma cidade usando coordenadas geográficas públicas.
    Use esta ferramenta sempre que o usuário perguntar sobre a temperatura ou clima de um local.
    """
    try:
        # Pegar Latitude e Longitude da cidade
        geo_url = f"https://open-meteo.com{city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()

        if not geo_res.get("results"):
            return f"Não encontrei as coordenadas para a cidade: {city}"

        location = geo_res["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        city_name = location["name"]

        # Busca a temperatura atual na API Open-Meteo
        weather_url = f"https://open-meteo.com{lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url).json()
        current_temp = weather_res["current_weather"]["temperature"]

        return f"O clima atual em {city_name} é de {current_temp}°C."
    except Exception as e:
        return f"Erro ao consultar o serviço de clima: {str(e)}"


@tool
def calculate_stock_price_with_tax(ticker: str, price: float, tax_percentage: float) -> str:
    """Calcula o valor final de compra de uma ação somando a porcentagem de taxa/corretagem.
    Use para responder perguntas financeiras que envolvam cálculos de compra de ativos.
    """
    final_price = price * (1 + (tax_percentage / 100))
    return f"O preço final calculado para o ativo {ticker.upper()} com {tax_percentage}% de taxa é R$ {final_price:.2f}."


# Agrupar tools em um dicionário para execução dinâmica posterior
tools_registry = {
    "get_current_weather": get_current_weather,
    "calculate_stock_price_with_tax": calculate_stock_price_with_tax
}

# Incializa o chat com modelo
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# CONEXÃO CRUCIAL: Vincula as ferramentas nativamente ao modelo OpenAI
llm_with_tools = llm.bind_tools(list(tools_registry.values()))

# Prompt do Sistema
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um chat inteligente que responde perguntas de vários assuntos. "
               "Você tem acesso a ferramentas para responder perguntas em tempo real. "
               "Se o usuário pedir algo que exija uma ferramenta, chame-a. Caso contrário, responda normalmente."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Definição da chain
chain = prompt | llm_with_tools

# Conexão com o Redis
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True)

SESSION_ID = "chat_com_ferramentas"
REDIS_KEY = f"chat_history:{SESSION_ID}"


def get_redis_history_raw(key: str) -> list:
    raw_messages = redis_client.lrange(key, 0, -1)
    langchain_messages = []
    for msg_str in raw_messages:
        try:
            msg_data = json.loads(msg_str)
            if msg_data["type"] == "human":
                langchain_messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                # Recupera mensagens de IA antigas mantendo metadados de tool_calls se existirem
                langchain_messages.append(
                    AIMessage(content=msg_data["content"], tool_calls=msg_data.get("tool_calls", [])))
            elif msg_data["type"] == "tool":
                langchain_messages.append(
                    ToolMessage(content=msg_data["content"], tool_call_id=msg_data["tool_call_id"]))
        except Exception as e:
            print(f"[ERRO PARSE]: {e}")
    return langchain_messages


def save_message_to_redis(key: str, role: str, content: str, tool_call_id: str = None, tool_calls: list = None):
    data = {"type": role, "content": content}
    if tool_call_id:
        data["tool_call_id"] = tool_call_id
    if tool_calls:
        data["tool_calls"] = tool_calls
    redis_client.rpush(key, json.dumps(data, ensure_ascii=False))
    redis_client.expire(key, 86400)


print("🤖 CLI CHAT com Ferramentas em Produção Iniciado!")
print("Experimente perguntar: 'Qual a temperatura atual no Rio de Janeiro?'\n")


while True:
    try:
        user_input = input("Você: ")
        if user_input.lower().strip() == 'sair':
            break
        if not user_input.strip():
            continue

        # Recupera o histórico do Redis
        history_for_context = get_redis_history_raw(REDIS_KEY)

        # Chamada ao LLM: decide se precisa de uma ferramenta ou se responde direto
        response = chain.invoke({
            "history": history_for_context,
            "input": user_input
        })

        # Salva o input inicial do usuário no Redis
        save_message_to_redis(REDIS_KEY, "human", user_input)

        # LOOP DE EXECUÇÃO DE TOOLS: Enquanto o modelo decidir que precisa chamar tool
        while response.tool_calls:
            # Salva o pensamento da IA (a intenção de chamar tool) no histórico
            save_message_to_redis(REDIS_KEY, "ai", response.content, tool_calls=response.tool_calls)

            history_for_context = get_redis_history_raw(REDIS_KEY)

            # Obter informações das tools
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"Executando tool '{tool_name}' com argumentos {tool_args}...")

                # Executa a função Python real buscando no nosso registro
                target_tool = tools_registry[tool_name]
                tool_output = target_tool.invoke(tool_args)

                print(f"Retorno da tool: {tool_output}")

                # Salva o resultado da tool (ToolMessage) no Redis
                save_message_to_redis(REDIS_KEY, "tool", tool_output, tool_call_id=tool_id)

            # Atualiza o contexto do histórico com os novos resultados das tools
            history_for_context = get_redis_history_raw(REDIS_KEY)

            # Chamada ao LLM: Envia os resultados das tools para gerar a resposta final em texto
            response = chain.invoke({
                "history": history_for_context,
                "input": user_input
            })

        # Salva a resposta final em formato de texto da IA no banco
        save_message_to_redis(REDIS_KEY, "ai", response.content)
        print(f"IA: {response.content}\n")

    except KeyboardInterrupt:
        break
