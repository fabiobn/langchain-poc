import os
import json
from dotenv import load_dotenv
import redis
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Incializa o chat com modelo
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# 3. Definição do Prompt (Template do Chat)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um engenheiro de software sênior prestativo e focado em boas práticas."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Criação da chain
chain = prompt | llm

# Conexão Direta com o Redis do Docker com decode para string
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True)
SESSION_ID = "usuario_engenharia"
REDIS_KEY = f"chat_history:{SESSION_ID}"


def get_redis_history_raw(key: str) -> list:
    """Busca o histórico do Redis e converte explicitamente para a lista que o LangChain exige."""
    raw_messages = redis_client.lrange(key, 0, -1)
    langchain_messages = []

    for msg_str in raw_messages:
        try:
            msg_data = json.loads(msg_str)
            if msg_data["type"] == "human":
                langchain_messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                langchain_messages.append(AIMessage(content=msg_data["content"]))
        except Exception as e:
            print(f"Erro ao decodificar mensagem do Redis: {e}")

    return langchain_messages


def save_message_to_redis(key: str, role: str, content: str):
    """Salva uma nova mensagem no formato JSON dentro da lista do Redis."""
    msg_json = json.dumps({"type": role, "content": content}, ensure_ascii=False)
    redis_client.rpush(key, msg_json)
    redis_client.expire(key, 86400) # Mantém por 24h


print("🤖 Chat Iniciado !")
print("Digite 'sair' para encerrar.\n")


while True:
    try:
        user_input = input("Você: ")
        if user_input.lower().strip() == 'sair':
            print("🤖 Encerrando o chat. Até logo!")
            break

        if not user_input.strip():
            continue

        # Recupera o histórico atualizado do Docker Redis
        current_history = get_redis_history_raw(REDIS_KEY)

        # Executa a cadeia passando o histórico atual e o novo input do usuário
        response = chain.invoke({
            "history": current_history,
            "input": user_input
        })

        # Commita as novas mensagens no banco
        save_message_to_redis(REDIS_KEY, "human", user_input)
        save_message_to_redis(REDIS_KEY, "ai", response.content)

        print(f"IA: {response.content}\n")

    except KeyboardInterrupt:
        print("\n🤖 Chat encerrado.")
        break
