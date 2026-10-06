import sys

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.investopedia.com/terms/c/covariance.asp"
loader = WebBaseLoader(url)
docs = loader.load()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(docs[0].page_content)  # Print the first 500 characters of the loaded content