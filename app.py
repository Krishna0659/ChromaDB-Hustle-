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
    {"id" : "doc1" , "text" : "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models (LLMs). It retrieves relevant documents from an external knowledge base and provides them as context to the LLM, enabling responses grounded in specific information rather than relying solely on pretrained knowledge."},
    {"id" : "doc2" , "text" : "RAG retrieves relevant information from a knowledge base using semantic similarity, keyword search, or metadata filters. The retrieved documents are passed to an LLM as context, allowing it to generate answers based on relevant information. This approach improves knowledge access without requiring model retraining."},
    {"id" : "doc3" , "text" : "Metadata filtering narrows document retrieval using structured attributes such as category, author, date, topic, or source. In a RAG pipeline, filters restrict the search space before or during vector retrieval, helping retrieve context that satisfies specific conditions and improving the relevance of the information provided to the LLM."}
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=doc["text"])

query = "Hello, world ! Where Do I study ?" 

results = collection.query(
    query_texts=[query],
    n_results=3
)

for doc_id, document, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0]
):
    similarity = 1 - distance

    print(f"ID: {doc_id}")
    print(f"Document: {document}")
    print(f"Cosine Distance: {distance:.4f}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print("-" * 40)


