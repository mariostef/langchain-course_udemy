import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

print("Initializing components...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    temperature=0,
)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],embedding=embeddings
)

retriever= vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template=ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question : {question}
    
    Provide detailed answer: 
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL 
    Returns a chain that can be invoked with {"question":"..."}
    """
    retrieval_chain=(
        RunnablePassthrough.assign( #the input is going to be the question
            context=itemgetter("question") | retriever | format_docs
        ) 
        |prompt_template
        | llm
        | StrOutputParser() # TO ACCESS THE .CONTENT KEY
    )
    return retrieval_chain

if __name__ == '__main__':
    print("Retrieving...")

    #Query
    query="What is pinecone in machine learning?"

    chain_with_lcel=create_retrieval_chain_with_lcel()
    result_with_lcel=chain_with_lcel.invoke({"question": query})
    print(f"The answer is {result_with_lcel}")
