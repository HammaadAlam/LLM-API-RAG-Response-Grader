import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load Environment
load_dotenv('.env')

def chroma_db():
    """
    Sets up the chroma database and stores the context chunks
    from the SQuAD2.0 Dev Set v2.0 into a collection
    """

    # Load SQuAD2.0 dataset
    with open("dev-v2.0.json", 'r') as f:
        data = json.load(f)

    # Establish client
    chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

    # Initialize embedding functions to generate vector representations of the text
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # Check if collection exists, otherwise create a new one
    if "squad2.0_contexts" in [collection.name for collection in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name="squad2.0_contexts")
    else:
        collection = chroma_client.create_collection(
            name="squad2.0_contexts",
            embedding_function=openai_ef
        )

    # Initialize counter for ids
    id_counter = 0

    # Extract context chunks and store in Chroma
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_id = str(id_counter)
            metadatas = ({'article_title': article['title']})
            collection.add(
                documents=[context],
                ids=[context_id],
                metadatas=[metadatas]
            )
            # Increment the counter
            id_counter += 1
            print(f"Context Number: {id_counter}")

    print(f"Successfully stored {id_counter} context chunks.")

chroma_db()