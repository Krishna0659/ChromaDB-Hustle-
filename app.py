import chromadb 
chroma_client = chromadb.Client()

collection_name = "test-collection"

collection = chroma_client.get_or_create_collection(
    collection_name , 
    configuration = {
        "hnsw" : {
            "space" : "cosine"
        }
    }
)

documents = [
    {"id" : "doc1" , "text" : "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models (LLMs). It retrieves relevant documents from an external knowledge base and provides them as context to the LLM, enabling responses grounded in specific information rather than relying solely on pretrained knowledge.", "metadata" : {
        "category" : "RAG",
        "topic" : "Fundamentals",
        "source" : "Learning Notes",
        "year" : 2026
    }},
    {"id" : "doc2" , "text" : "RAG retrieves relevant information from a knowledge base using semantic similarity, keyword search, or metadata filters. The retrieved documents are passed to an LLM as context, allowing it to generate answers based on relevant information. This approach improves knowledge access without requiring model retraining.", "metadata" : {
        "category" : "RAG",
        "topic" : "Retrieval Process",
        "source" : "Learning Notes",
        "year" : 2026
    }},
    {"id" : "doc3" , "text" : "Metadata filtering narrows document retrieval using structured attributes such as category, author, date, topic, or source. In a RAG pipeline, filters restrict the search space before or during vector retrieval, helping retrieve context that satisfies specific conditions and improving the relevance of the information provided to the LLM.", "metadata" : {
        "category" : "RAG",
        "topic" : "Metadata Filtering",
        "source" : "Learning Notes",
        "year" : 2026
    }}
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], metadatas=[doc["metadata"]])


query = "How does metadata filtering help in RAG ?" 

results = collection.query(
    query_texts=[query],
    n_results=1,
    where = {
        "$and" : [
            {"category" : "RAG"},
            {"topic" : "Metadata Filtering"},
            {"year" : {"$gte" : 2026}}
        ]
    }
)

print(f"\nQuery: {query}")
print("=" * 70)

for i, doc_id in enumerate(results["ids"][0]):

    document = results["documents"][0][i]
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    # Convert squared L2 distance into a bounded custom score
    similarity_score = 1 / (1 + distance)

    print(f"\nRank: {i + 1}")
    print(f"ID: {doc_id}")
    print(f"Document: {document}")
    print(f"Metadata: {metadata}")
    print(f"Distance: {distance:.4f}")
    print(f"Similarity Score: {similarity_score:.2%}")
    print("-" * 70)