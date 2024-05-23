# main.py

# Import necessary modules
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as pns
from langchain.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Set up Pinecone index
api_key = os.getenv("PINECONE_API_KEY")
index_name = "langchain-demo-2"
dimension = 768
metric = "cosine"
cloud = "aws"
region = "us-east-1"

# Define function to create Pinecone index
def create_pinecone_index(api_key, index_name, dimension, metric, cloud, region):
    try:
        pinecone_client = Pinecone(api_key=api_key)
        print("Connection established successfully.")

        if index_name not in pinecone_client.list_indexes().names():
            spec = ServerlessSpec(cloud=cloud, region=region)
            pinecone_client.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")

    except Exception as e:
        print(f"An error occurred while creating the Pinecone index: {e}")

# Create Pinecone index
#create_pinecone_index(api_key, index_name, dimension, metric, cloud, region)


# Set up embeddings
embeddings = HuggingFaceEmbeddings()

# Set up Pinecone retriever
docsearch = pns.from_existing_index(index_name, embeddings)

# Set up language model
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_k": 50})

# Set up prompt template
template = """
Your job is to use our company's knowledge base and answer questions based on that.
If you don't know the answer, just say you don't know.
Keep the answer within 2 sentences and concise.


Context: {context}
Question: {question}
Answer: 
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Define the RAG pipeline
rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
