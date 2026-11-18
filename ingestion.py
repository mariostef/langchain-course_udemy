import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

    loader=UnstructuredLoader(file_path=r"C:\Users\MariosStefanis\Desktop\langchain-rag\mediumblog1.txt",chunking_strategy="basic",max_characters=1000000)
    document=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Creating embeddings and uploading to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
    )
    print("Ingestion complete")

