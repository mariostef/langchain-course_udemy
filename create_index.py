import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name=os.environ["INDEX_NAME"]

    if pc.has_index(index_name):
        index_info = pc.describe_index(index_name)
        if index_info.dimension != 384 or index_info.metric != "cosine":
            raise ValueError("The existing index must use dimension=384 and metric=cosine. Choose a new INDEX_NAME.")
        print(f"Index {index_name} already exists")
    else:
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        print(f"The index {index_name} was created")

