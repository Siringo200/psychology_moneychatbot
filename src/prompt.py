from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

system_prompt = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, just say that you don't know. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)