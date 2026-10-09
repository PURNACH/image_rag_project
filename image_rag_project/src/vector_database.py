import chromadb

from .embeddings import create_embeddings


# Persistent Chroma database
client = chromadb.PersistentClient(
    path="./chroma_db"
)


COLLECTION_NAME = "image_rag_collection"


def get_collection():
    """
    Get or create ChromaDB collection.
    """

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def upsert_chunks(chunks):
    """
    Store text chunks and embeddings in ChromaDB.
    """

    if not chunks:
        return

    collection = get_collection()

    embeddings = create_embeddings(chunks)

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


def search_chunks(query, top_k=3):
    """
    Search ChromaDB for relevant chunks.
    """

    collection = get_collection()

    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])

    if not documents:
        return []

    return documents[0]