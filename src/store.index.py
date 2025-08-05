from src.helper import load_pdf_file, text_splitter, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data='Data')
text_chunks = text_splitter(extracted_data)
langchain_embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "moneychatbot2"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for all-MiniLM-L6-v2 model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=langchain_embeddings
)