import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# Incializa o chat com modelo
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Definição do Prompt (Template do Chat)
# O MessagesPlaceholder é onde  histórico de mensagens guardado será injetado dinamicamente
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um engenheiro de software sênior prestativo e focado em boas práticas."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Criação da Cadeia (Chain) usando LCEL: prompt + modelo
chain = prompt | llm

# Gerenciamento de Persistência Checkpointer Simples
# Memória para guardar as sessões
session_store = {}

# Função para obter o histório referente a uma sessão
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Busca ou cria um histórico persistente para o ID fornecido."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Configuração da sessão atual (Simula um usuário logado)
SESSION_ID = "usuario_engenharia"

print("🤖 Chat Iniciado! Digite 'sair' para encerrar a conversa.\n")

while True:
    try:
        user_input = input("Você: ")
        if user_input.lower().strip() == 'sair':
            print("🤖 Encerrando o chat. Até logo!")
            break

        if not user_input.strip():
            continue

        # Recupera o histórico salvo desta sessão
        history_backend = get_chat_history(SESSION_ID)

        # Executa a cadeia passando o histórico atual e o novo input do usuário
        response = chain.invoke({
            "history": history_backend.messages,
            "input": user_input
        })

        # Atualiza o histórico(checkpointer) utilizando os métodos de InMemoryChatMessageHistory
        history_backend.add_user_message(user_input)
        history_backend.add_ai_message(response.content)

        # Exibe a resposta do modelo
        print(f"IA: {response.content}\n")

    except KeyboardInterrupt:
        print("\n🤖 Chat encerrado abruptamente.")
        break
