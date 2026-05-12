from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

response = llm.invoke("Hello")

print(response)