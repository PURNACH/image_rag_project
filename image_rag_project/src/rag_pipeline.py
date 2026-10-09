from .vector_store import search_chunks


def retrieve_context(question, top_k=3):
    """
    Retrieve relevant text chunks from ChromaDB.
    """

    documents = search_chunks(
        query=question,
        top_k=top_k
    )

    if not documents:
        return ""

    context = "\n\n".join(documents)

    return context