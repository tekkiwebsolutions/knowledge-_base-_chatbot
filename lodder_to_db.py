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

class TextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                documents = file.read()  # Read the entire file as a single string
            return documents
        except FileNotFoundError:
            print(f"File '{self.file_path}' not found.")
            return ""
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return ""

# Load environment variables
load_dotenv()

# Set up Pinecone index
api_key = os.getenv("PINECONE_API_KEY")
index_name = "langchain-demo-2"
dimension = 768
metric = "cosine"
cloud = "aws"
region = "us-east-1"

loader = TextLoader('data/data3.txt')# insert data here
documents = loader.load()

# Set up text splitter and split documents
text_splitter = CharacterTextSplitter(chunk_size=5578, chunk_overlap=4)

# Split the text into smaller chunks directly
docs = [documents[i:i+1000] for i in range(0, len(documents), 1000)]

# Set up embeddings
embeddings = HuggingFaceEmbeddings()

def insert_data_to_pinecone(embeddings, docs):
    print("Inserting data into Pinecone index...")
    try:
        print("Existing Pinecone index retrieved successfully.")

        data_to_insert = []
        for doc_id, doc in enumerate(docs):
            doc = [str(part) for part in doc]
            embedding = embeddings.embed_documents(doc)
            embedding = [float(value) for sublist in embedding for value in sublist]
            data_to_insert.append({"id": str(doc_id), "values": embedding})

        # Initialize Pinecone client with API key inside the function
        pinecone_client = Pinecone(api_key=api_key)
        index = pinecone_client.Index(index_name)
        index.upsert(vectors=data_to_insert)
        print("Data inserted into the Pinecone index successfully.")

    except Exception as e:
        print(f"An error occurred while inserting data to the Pinecone index: {e}")

insert_data_to_pinecone(embeddings,docs)