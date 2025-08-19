from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

loader =  WebBaseLoader('https://en.wikipedia.org/wiki/Nepal_national_cricket_team#')

docs=loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = Chroma(collection_name='webpage', embedding_function=embeddings, persist_directory='chroma')
vector_store.add_documents(chunks)

prompt = ChatPromptTemplate([
    ("system", "You are a helpful ai assistant who answers queries provided by users from only the following context and answer i don't have the context to your query if no relevant context to user's query is found. Context:{context}"),
    ("human", "Input: {input}")
])
model = ChatGroq(model_name='deepseek-r1-distill-llama-70b', reasoning_format='hidden')

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)

response = retrieval_chain.invoke({'input': 'What are the major achievements?'})
print(response['answer'])