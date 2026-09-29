import os
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

1. `download_and_save_file`: Downloads a URL to a local file. Use this first.
2. `count_substring_lines_in_file`: Counts how many lines contain a specific substring in the saved file.
3. `find_first_occurrence_line_number`: Finds the 1-based line number of the first line containing a substring.
4. `read_file_sample`: Reads a specific range of lines from the file (useful for context/synopsis).

Do not guess line counts or positions—ground them strictly in tool results from the saved file. 
If tools fail, use `null` and report the error."""

LOCAL_FILE_PATH = "../../gutenberg_book.txt"


@tool
def download_and_save_file(url: str) -> str:
    """Downloads a document from a URL and saves it locally to save tokens."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="replace")
        with open(LOCAL_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        return f"Success: File downloaded and saved locally to {LOCAL_FILE_PATH}."
    except urllib.error.URLError as e:
        return f"Download failed: {e}"


@tool
def count_substring_lines_in_file(substring: str) -> str:
    """Counts how many lines in the saved file contain the exact substring."""
    if not os.path.exists(LOCAL_FILE_PATH):
        return "Error: File not downloaded yet. Call download_and_save_file first."

    count = 0
    with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if substring in line:
                count += 1
    return f"The substring '{substring}' appears in {count} unique lines."


@tool
def find_first_occurrence_line_number(substring: str) -> str:
    """Finds the 1-based line number of the first line that contains the substring."""
    if not os.path.exists(LOCAL_FILE_PATH):
        return "Error: File not downloaded yet. Call download_and_save_file first."

    with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if substring in line:
                return f"The first occurrence of '{substring}' is at line {idx}."
    return f"Substring '{substring}' not found in the file."


@tool
def read_file_sample(start_line: int, num_lines: int) -> str:
    """Reads a small chunk of lines (e.g., from line 100 to 150) to understand the content without hitting token limits."""
    if not os.path.exists(LOCAL_FILE_PATH):
        return "Error: File not downloaded yet. Call download_and_save_file first."

    lines = []
    with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx >= start_line:
                lines.append(f"{idx}: {line.strip()}")
            if len(lines) >= num_lines:
                break
    return "\n".join(lines)


# Inicializando o modelo (Mudei para o Llama 3.3 que lida melhor com múltiplos passos de ferramentas no Groq)
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0.1,  # Menor temperatura ajuda a seguir instruções rígidas de dados
    timeout=600,
)

checkpointer = InMemorySaver()

ferramentas = [
    download_and_save_file,
    count_substring_lines_in_file,
    find_first_occurrence_line_number,
    read_file_sample
]

agent = create_agent(
    model=model,
    tools=ferramentas,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"""Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer as much as you can using your available tools:

1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `Daisy`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was."""

print("Running create_agent...", flush=True)
agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)

print("\ncreate_agent:")
# Exibindo o texto final gerado pelo agente de forma limpa
print(agent_result["messages"][-1].content)
