import os
import json
import sys
from dotenv import load_dotenv
import redis
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Incializa o chat com modelo
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Definição do Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um engenheiro de software sênior prestativo e focado em boas práticas."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Criação da chain
chain = prompt | llm

# Conexão Estabilizada com o Redis
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True  # Garante o parse de strings direto (sem quebra de .decode)
)

# APLICANDO A JANELA DESLIZANTE
# Ex: 6 mensagens = últimas 3 interações completas (User + AI)
# LIMITE_JANELA_CONTEXTO = 6

# APLICANDO A JANELA DESLIZANTE
# def get_redis_history_sliding_window(key: str, limit: int) -> list:
def get_redis_history_raw(key: str) -> list:
    """Busca a lista do Redis e monta os objetos nativos do LangChain."""
    raw_messages = redis_client.lrange(key, 0, -1)

    # APLICANDO A JANELA DESLIZANTE: Pegamos apenas as últimas 'limit' strings
    # if limit > 0:
    #     raw_messages = raw_messages[-limit:]

    langchain_messages = []
    for msg_str in raw_messages:
        try:
            msg_data = json.loads(msg_str)
            if msg_data["type"] == "human":
                langchain_messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                langchain_messages.append(AIMessage(content=msg_data["content"]))
        except Exception as e:
            print(f"[ERRO PARSE]: {e}")

    return langchain_messages


def save_message_to_redis(key: str, role: str, content: str):
    """Salva uma mensagem na lista e renova o tempo de vida dela por 24h."""
    msg_json = json.dumps({"type": role, "content": content}, ensure_ascii=False)
    redis_client.rpush(key, msg_json)
    redis_client.expire(key, 86400)


# Inicialização com Múltiplas Sessões Dinâmicas
print("==================================================")
print("🤖 CLI CHAT - MULTI-SESSÕES & STREAMING ATIVOS")
print("==================================================")

# Definição do usuário
session_id = input("Identificador de Usuário: ").strip().lower()
if not session_id:
    session_id = "default_user"

# Geração dinâmica da chave no Redis
REDIS_KEY = f"chat_history:{session_id}"

print(f"\nSessão ativa vinculada à chave: '{REDIS_KEY}'")
print("Digite 'sair' para encerrar a conversa.\n")

# Loop Interativo (CLI)
while True:
    try:
        user_input = input(f"[{session_id}] Você: ")
        if user_input.lower().strip() == 'sair':
            print(f"🤖 Encerrando a sessão de {session_id}. Até logo!")
            break

        if not user_input.strip():
            continue

        # Obter histórico relacionado ao usuário da sessão
        history_for_context = get_redis_history_raw(REDIS_KEY)
        # APLICANDO A JANELA DESLIZANTE
        #history_for_context = get_redis_history_sliding_window(REDIS_KEY, LIMITE_JANELA_CONTEXTO)

        # Execução do LLM usando .stream() para exibir a resposta palavra por palavra
        print("IA: ", end="")
        full_response_content = ""

        # O método .stream() devolve pequenos blocos (chunks) da resposta à medida que saem da API
        response_chunked = chain.stream({
            "history": history_for_context,
            "input": user_input
        })
        for chunk in response_chunked:
            token = chunk.content
            full_response_content += token

            # Printa o token sem quebra de linha automática
            sys.stdout.write(token)
            # Força o terminal a renderizar o caractere imediatamente na tela
            sys.stdout.flush()

        print("\n")  # Força uma quebra de linha ao finalizar a resposta

        # Salva mensagens no Redis para histórico
        save_message_to_redis(REDIS_KEY, "human", user_input)
        save_message_to_redis(REDIS_KEY, "ai", full_response_content)

    except KeyboardInterrupt:
        print(f"\n🤖 Chat de {session_id} encerrado abruptamente.")
        break
