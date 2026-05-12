from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/papers/1706.03762v7.pdf")
documents = loader.load()

print(documents[0].page_content[:1000])